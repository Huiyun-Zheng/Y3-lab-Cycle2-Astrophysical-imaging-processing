"""Review original detections using pixel checks"""

from pathlib import Path
from collections import Counter
import hashlib
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
INPUT = BASE / "results_before_refinement"
OUTPUT = BASE / "results"

# These are image-quality limits, not measured neighbour-light fractions.
MAX_LABEL_AREA = 0.03
MAX_LABEL_LIGHT = 0.02
MAX_SHARED_LIGHT = 0.02
# Check beyond the aperture edge without enlarging the photometry aperture.
CLOSE_PAIR_DISTANCE = 1.25
BRIDGE_MINIMUM_SNR = 3.0
PAIR_PEAK_MINIMUM_SNR = 5.0
MINIMUM_SNR = 5.0
RIGHT_STRIP_START = 2450
MASK_PADDING = 4


def aperture_grid(row, shape, scale=1.0):
    """
    Recreate an elliptical aperture using the position, size, and angle stored in the origina catalogue.
    """
    x, y = float(row["x"]), float(row["y"])
    a, b = scale * row["aperture_a_pix"], scale * row["aperture_b_pix"]
    angle = np.radians(row["aperture_theta_deg"])

    # Calculate the sine and cosine for the rotation.
    c, s = np.cos(angle), np.sin(angle)
    # Find how far the rotated ellipse extends in x and y.
    ex, ey = np.hypot(a * c, b * s), np.hypot(a * s, b * c)

    # Create a rectangle around the ellipse.
    # Keep the range inside the image boundaries.
    x0, x1 = max(0, int(np.floor(x - ex))), min(shape[1], int(np.ceil(x + ex)) + 1)
    y0, y1 = max(0, int(np.floor(y - ey))), min(shape[0], int(np.ceil(y + ey)) + 1)

    yy, xx = np.mgrid[y0:y1, x0:x1]

    # Shift the coordinates to the object centre and rotate them
    # so that the ellipse is aligned with the new axes.
    u, v = (xx - x) * c + (yy - y) * s, -(xx - x) * s + (yy - y) * c

    # Mark pixels whose centres fall inside the ellipse.
    inside = (u / a)**2 + (v / b)**2 <= 1
    # Check whether the full elipse reaches the image boundary.
    edge = x - ex < 0 or y - ey < 0 or x + ex >= shape[1] or y+ey >= shape[0]

    return (slice(y0, y1), slice(x0, x1)), xx, yy, inside, edge


def internal_double(data, rms, own):
    """Check whether an object contains two clearly separated brightness peaks with a significant dip between them."""
    smooth = ndi.gaussian_filter(data.astype(float), 0.7)

    # Find local brightness peaks within the current object.
    # Each peak must be the brightese in its 5*5 neighbourhood
    # and have a brightness greater than eight times the local noise.
    peaks = own & (smooth == ndi.maximum_filter(smooth, 5)) & (smooth > 8*rms)
    # Give each connected peak region a separate label.
    groups, n = ndi.label(peaks)

    if n < 2:
        return False

    # FIne the brightest pixel in each labelled peak region.
    points = ndi.maximum_position(smooth, groups, np.arange(1,n + 1))
    # Sort the peaks from brightest to fainest.
    points = sorted(points, key=lambda p: smooth[p], reverse=True)

    # Use the brightest peak as the main reference peak.
    primary = points[0]

    # Check each of the remaining peaks.
    for p in points[1:]:
        distance = np.linalg.norm(np.subtract(p, primary))  # distance between this peak and the main peak

        # Ignore peaks that are too close or much fainter than the main peak.
        if distance < 5 or smooth[p] < 0.2 * smooth[primary]:
            continue

        # Create points along the straight line between the peaks.
        # Sample the line at roughly half-pixel intervals.
        line = np.linspace(primary, p, max(3, int(2 * distance) + 1)).T

        # Check whether the brightness drop is large enough.
        valley = ndi.map_coordinates(smooth, line, order=1).min()
        if smooth[p] - valley > max(2 * float(np.median(rms[own])), 0.2 * smooth[p]):
            return True

    return False


def close_pair_bridge(first, second, signal, rms, bad):
    """
    Check whether two nearby sources are connected by significant light.
    Only check sources that came from the same original object.
    """

    # Only check sources from the same original detected region.
    if first["parent_id"] != second["parent_id"]:
        return False, np.nan, np.nan

    # Find the distance and direction between the source centres.
    dx, dy = second["x"] - first["x"], second["y"] - first["y"]
    direction = np.arctan2(dy,dx)

    # Fine the radius of each ellipse along the line joining the centres.
    radii = []
    for row in (first, second):
        angle = direction - np.radians(row["aperture_theta_deg"])
        radii.append(1 / np.sqrt((np.cos(angle) / row["aperture_a_pix"])**2 + (np.sin(angle)/row["aperture_b_pix"])**2))

    separation = np.hypot(dx,dy)  # centre separation
    # Compare the separation with the combined ellipse radii.
    relative_distance = separation / sum(radii)

    # Ignore sources that are too far apart
    if relative_distance > CLOSE_PAIR_DISTANCE:
        return False, relative_distance, np.nan

    # Get the brightest pixel position of each position.
    peaks = np.array([[first["peak_y"],first["peak_x"]], [second["peak_y"],second["peak_x"]]], dtype=int)
    # Get the signal and noise at both brightness peaks.
    peak_signal = signal[peaks[:, 0], peaks[:, 1]]
    peak_noise = rms[peaks[:, 0], peaks[:, 1]]

    # Require both peaks to have a valid and sufficiently high SNR.
    if not np.all(np.isfinite(peak_signal / peak_noise)) or np.min(peak_signal / peak_noise) < PAIR_PEAK_MINIMUM_SNR:
        return False, relative_distance, np.nan

    # Calculate the distance between the two brightness peaks.
    length = np.linalg.norm(peaks[1] - peaks[0])
    # Sample the line between the peaks at about half-pixel intervals.
    line = np.linspace(peaks[0], peaks[1], max(5, int(np.ceil(2 * length)) + 1)).T
    # Round the sample positions to the nearest pixels.
    line_pixels = np.rint(line).astype(int)

    # Skip the pair if the line passes through any bad pixels.
    if np.any(bad[line_pixels[0], line_pixels[1]]):
        return False, relative_distance, np.nan

    # Get the signal slong the line using linear interpolation.
    values = ndi.map_coordinates(signal, line, order=1)
    # Get the noise RMS along the same line.
    errors = ndi.map_coordinates(rms, line, order=1)
    # Fine the lowest SNR between the two peaks.
    bridge = float(np.min(values/errors))

    # FLag the pair if the entire sampled line has sufficient SNR.
    # Also return the relative distance and minimum SNR.
    return bool(np.isfinite(bridge) and bridge >= BRIDGE_MINIMUM_SNR), relative_distance, bridge


def draw_apertures(ax, table, old=False):
    if old:
        selected = table["original_usable"]
    else:
        selected = table["usable"]

    for use, colour in [(True, "blue"), (False, "red")]:
        patches = [Ellipse((r["x"], r["y"]), 2 * r["aperture_a_pix"],
                           2 * r["aperture_b_pix"], angle=r["aperture_theta_deg"])
                               for r in table[selected == use]]
        ax.add_collection(PatchCollection(patches, facecolors="none",
                          edgecolors=colour, linewidths=0.55), autolim=False)


def save_plots(table, signal, rms, bad):
    display = signal / rms
    fig, axes = plt.subplots(1, 2, figsize=(13,11))
    for ax in axes:
        ax.imshow(display, origin="lower", cmap="gray", vmin=-1, vmax=12)
        # Show the full exclusion mask on both panels, below the apertures.
        ax.imshow(np.ma.masked_where(~bad, np.ones(bad.shape)), origin="lower", cmap="autumn", vmin=0, vmax=1, alpha=0.4)
        ax.set_xlabel("x / pixel")
        ax.set_ylabel("y / pixel")
    axes[0].set_title("Masks, including right strip and fixed defect")
    draw_apertures(axes[1], table)
    axes[1].set_title(f"{len(table)} candidates; {table["usable"].sum()} usable\nBlue: usable; red: excluded")
    fig.tight_layout()
    fig.savefig(OUTPUT / "detection_overview.png", dpi=300)
    plt.close(fig)

    # Show the 12 highest-peak recovered sources, with nearby apertures included.
    recovered = table[table["recovered"]]
    chosen = recovered[np.argsort(recovered["peak_adu"])[::-1]][:12]
    if len(chosen):
        fig, axes = plt.subplots(len(chosen), 2, figsize=(7, 3 * len(chosen)), squeeze=False)
        for pair, row in zip(axes, chosen):
            half = max(28, int(2 * row["aperture_a_pix"]))
            neighbours = table[(np.abs(table["x"] - row["x"]) < half + 35) & (np.abs(table["y"] - row["y"]) < half + 35)]
            x0,x1 = max(0, int(row["x"]) - half), min(signal.shape[1],int(row["x"]) + half + 1)
            y0,y1 = max(0,int(row["y"]) - half), min(signal.shape[0],int(row["y"]) + half + 1)
            for ax, old in zip(pair,[True, False]):
                ax.imshow(display[y0:y1,x0:x1], origin="lower", cmap="gray", vmin=-1, vmax=12, extent=(x0-0.5,x1-0.5,y0-0.5,y1-0.5))
                draw_apertures(ax,neighbours,old)
                ax.set_xlim(x0,x1)
                ax.set_ylim(y0,y1)
                ax.set_title(f"ID {row["id"]}: "+("original" if old else "reviewed"))
        fig.tight_layout()
        fig.savefig(OUTPUT / "recovered_bright_before_after.png", dpi=300)
        plt.close(fig)


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = Table.read(INPUT / "sources.ecsv")
    input_hash = hashlib.sha256((INPUT / "sources.ecsv").read_bytes()).hexdigest()
    image, header = fits.getdata(ROOT / "Astro" / "Fits_Data" / "mosaic.fits", header=True)
    gain = float(header["GAIN"])

    with fits.open(INPUT / "detection_products.fits") as hdul:
        background, rms = hdul["BACKGROUND"].data.copy(), hdul["RMS"].data.copy()
        labels = hdul["SOURCES"].data.copy()
        bad = hdul["DETMASK"].data.astype(bool)

    # Subtract the estimated background from the image.
    signal = image.astype(np.float32) - background

    # Mask the unwanted strip on the right side of the image.
    strip = np.zeros(image.shape, bool)
    strip[:, RIGHT_STRIP_START:] = True
    # Mask the manually identified cross-shaped defect.
    manual = np.zeros(image.shape, bool)
    manual[423:432,1026:1046] = True
    manual[428:453,1033:1041] = True
    # Expand the new masks ad add them to the detector mask.
    bad |= ndi.binary_dilation(strip | manual, iterations=MASK_PADDING)

    # Create new Boolean columns for different rejection reasons.
    n = len(table)
    for name in ["mask_rejected", "neighbour_core", "internal_double", "shared_light", "close_pair_bridge", "sky_unreliable"]:
        table[name] = np.zeros(n, bool)
    # Create new columns for contamination fraction, SNR and flux.
    for name in ["label_area_fraction", "label_light_fraction", "aperture_snr", "measured_aperture_flux_adu"]:
        table[name] = np.zeros(n, float)

    # Keep the original results for later comparison.
    table["original_usable"] = table["usable"].copy()
    table["original_flags"] = table["flags"].copy()
    table["original_aperture_flux_adu"] = table["aperture_flux_adu"].copy()

    pixels, totals = [], []
    peak_lookup = {int(r["segment_id"]):(int(r["peak_y"]), int(r["peak_x"])) for r in table}
    print("Checking apertures, neighbours and masks...", flush=True)
    for i, row in enumerate(table):
        bounds, xx, yy, ap, edge = aperture_grid(row, image.shape)
        data, noise, labs = signal[bounds], rms[bounds], labels[bounds]

        table["mask_rejected"][i] = edge or not ap.any() or np.any(bad[bounds][ap]) or not np.all(np.isfinite(data[ap]) & np.isfinite(noise[ap]) & (noise[ap] > 0))

        # Keep only positive signal for contamination calculations.
        positive = np.maximum(data, 0)
        total = max(float(positive[ap].sum()), 1e-30)
        pixels.append(yy[ap] * image.shape[1] + xx[ap])
        totals.append(total)

        # Find aperture pixels assigned to other detected sources
        other = ap & (labs > 0) & (labs != row["segment_id"])
        # Measure the fraction of aperture area belonging to other labels.
        table["label_area_fraction"][i] = other.sum() / max(ap.sum(), 1)
        # Measure the fraction of positive signal in those pixels.
        table["label_light_fraction"][i] = positive[other].sum() / total

        flux = float(data[ap].sum())
        error = np.sqrt(np.sum(noise[ap].astype(float)**2 + positive[ap] / gain))
        table["measured_aperture_flux_adu"][i] = flux
        table["aperture_snr"][i] = flux / error if error > 0 else np.nan

        core = np.zeros(ap.shape, bool)  # mask for bright pixels belonging to neighbours.
        # Check each neighbouring source found inside the aperture.
        for label in np.unique(labs[other]):
            py, px = peak_lookup[int(label)]
        # Check whether the neighbour's peak lies inside current aperture.
            if bounds[0].start <= py < bounds[0].stop and bounds[1].start <= px < bounds[1].stop:
                if ap[py - bounds[0].start, px - bounds[1].start]:
                    table["neighbour_core"][i] = True

        # Find bright pixels belonging to this neighbour.
            peak = signal[py, px]
            core |= other & (labs == label) & (data > 0.3 * peak) & (peak > 10 * noise)
        # Flag significant contamination from neighbouring bright cores.
        table["neighbour_core"][i] |= core.sum() >= 2 and positive[core].sum() / total > 0.01

        # Check for multiple brightness peaks within a valid source.
        if not table["mask_rejected"][i]:
            # Use a larger cutout to reduce smoothing edge effects.
            bb, _, _, outer, _ = aperture_grid(row, image.shape, 1.5)
            own = (labels[bb] == row["segment_id"]) & outer  # pixels belonging to this source.
            # Flag possible internal fouble-peak structure
            table["internal_double"][i] = internal_double(signal[bb], rms[bb], own)

        # Check local sky quality for soources near defects or boundaries.
        if row["near_artefact"] or row["near_boundary"]:
            # Create a region extending to twice the aperture size.
            bb, xx2, yy2, outer, _ = aperture_grid(row, image.shape, 2)

            t = np.radians(row["aperture_theta_deg"])
            dx = xx2 - row["x"]
            dy = yy2 - row["y"]
            radius = ((dx * np.cos(t) + dy * np.sin(t)) / row["aperture_a_pix"])**2 + ((-dx * np.sin(t) + dy * np.cos(t)) / row["aperture_b_pix"])**2
            ring = outer & (radius >= 1.5**2)  # elliptical sky ring between 1.5 and 2 radii
            clean = ring & ~bad[bb] & (labels[bb] == 0) & np.isfinite(signal[bb]) & (rms[bb]>0)
            # Reject sky estimates with too few clean pixels.
            table["sky_unreliable"][i] = clean.sum() < 40 or clean.sum() < 0.5 * ring.sum()
            # Check whether the local background has a large residual.
            if not table["sky_unreliable"][i]:
                table["sky_unreliable"][i] = abs(np.median(signal[bb][clean] / rms[bb][clean])) > 1
    print("Checking significant light shared by aperture pairs...", flush=True)

    xy = np.column_stack([table["x"], table["y"]])
    radii = np.asarray(table["aperture_a_pix"])
    pair_rows = []
    bridge_rows = []
    # Use a KDTree to find potentially nearby source pairs.
    for i,j in cKDTree(xy).query_pairs(2 * CLOSE_PAIR_DISTANCE * max(radii), output_type="ndarray"):
        # Check for significant light connecting the two peaks.
        close, relative_distance, bridge = close_pair_bridge(table[i], table[j], signal, rms, bad)
        # Record valid light-bridge measurements.
        if np.isfinite(bridge):
            bridge_rows.append((int(table["id"][i]), int(table["id"][j]), relative_distance, bridge, close))
        # Flag both sources if they have a significant light bridge.
        if close:
            table["close_pair_bridge"][[i,j]] = True
        # Skip pairs whose apertures cannot overlap.
        if np.linalg.norm(xy[i] - xy[j]) >= radii[i] + radii[j]:
            continue

        common = np.intersect1d(pixels[i], pixels[j], assume_unique=True)  # pixels shared by the two elliptical apertures.
        common = common[~bad.ravel()[common]]
        if len(common) < 3:
            continue
        # Get signal and noise values in the shared region.
        values = signal.ravel()[common]
        errors = rms.ravel()[common].astype(float)
        # Calculate the shared flux and its fraction of each aperture.
        flux = float(values.sum())
        f1, f2 = flux / totals[i], flux / totals[j]
        # Require significant shared light for both sources
        reject = (np.count_nonzero(values > 3 * errors) >= 3 and flux > 5 * np.sqrt(np.sum(errors**2)) and min(f1,f2) > MAX_SHARED_LIGHT)
        if reject:
            table["shared_light"][[i,j]] = True
        pair_rows.append((int(table["id"][i]), int(table["id"][j]), f1, f2, reject))

    # Collect all rejection regions for each source.
    flags = []
    for row in table:
        reasons = [name for name in ["mask_rejected", "neighbour_core", "internal_double", "shared_light", "close_pair_bridge", "sky_unreliable"] if row[name]]
        if row["aperture_capped"]:
            reasons.append("aperture_capped")
        if row["elongation"] > 6:
            reasons.append("very_elongated")
        if not np.isfinite(row["aperture_snr"]) or row["aperture_snr"] < MINIMUM_SNR:
            reasons.append("low_snr")
        if row["label_area_fraction"] > MAX_LABEL_AREA:
            reasons.append("label_area")
        if row["label_light_fraction"] > MAX_LABEL_LIGHT:
            reasons.append("label_light")
        flags.append("|".join(reasons) or "none")

    table["flags"] = np.asarray(flags, dtype="U250")
    # A source is usable only if it passes all quality checks.
    table["usable"] = table["flags"] == "none"
    table["recovered"] = table["usable"] & ~table["original_usable"]
    table["newly_excluded"] = ~table["usable"] & table["original_usable"]
    table["aperture_flux_adu"] = np.where(table["usable"], table["measured_aperture_flux_adu"], np.nan)

    # Record the input catalogue hash and review settings.
    table.meta["input_catalogue_sha256"] = input_hash
    table.meta["review"] = "Pixel-based quality review; no profile fitting. Original quality columns remain historical; use usable and flags for current selection."
    table.meta["flux_note"] = "Signed observed ADU in unchanged binary apertures; no neighbour subtraction. Positive light is used only for quality ratios and approximate shot noise."
    table.meta["snr_note"] = "Independent-pixel RMS plus positive signal/GAIN; correlated noise is not calibrated."
    table.meta["limits"] = f"label area {MAX_LABEL_AREA}; label light {MAX_LABEL_LIGHT}; shared light {MAX_SHARED_LIGHT}; SNR {MINIMUM_SNR}; close distance {CLOSE_PAIR_DISTANCE}; bridge SNR {BRIDGE_MINIMUM_SNR}; pair peak SNR {PAIR_PEAK_MINIMUM_SNR}"

    for name,t in [("sources", table), ("usable_sources", table[table["usable"]]), ("recovered_sources", table[table["recovered"]])]:
        t.write(OUTPUT / (name+".ecsv"), overwrite=True)
        t.write(OUTPUT / (name+".csv"), format="ascii.csv",overwrite=True)
    Table(rows=pair_rows,names=["id1", "id2", "shared_fraction1", "shared_fraction2", "rejected"], dtype=[int, int, float, float, bool]).write(OUTPUT/"shared_pairs.ecsv", overwrite=True)
    Table(rows=bridge_rows,names=["id1", "id2", "relative_distance", "minimum_bridge_snr", "rejected"], dtype=[int, int, float, float, bool]).write(OUTPUT/"close_pairs.ecsv", overwrite=True)

    # Open the original detection products to update the masks.
    with fits.open(INPUT / "detection_products.fits") as hdul:
        hdul["BADMASK"].data = (hdul["BADMASK"].data.astype(bool)|strip|manual).astype(np.uint8)
        hdul["DETMASK"].data = bad.astype(np.uint8)
        hdul["ARTEFACT"].data = (hdul["ARTEFACT"].data.astype(bool)|manual).astype(np.uint8)
        hdul["BOUNDARY"].data = (hdul["BOUNDARY"].data.astype(bool)|strip).astype(np.uint8)
        hdul.append(fits.ImageHDU(strip.astype(np.uint8), name="STRIPMASK"))
        hdul.append(fits.ImageHDU(manual.astype(np.uint8), name="MANUALMASK"))
        hdul.writeto(OUTPUT / "detection_products.fits", overwrite=True)

    # Fine the 100 sources with the highest peak brightness.
    top=np.argsort(table["peak_adu"])[-100:]
    # Summary how source usability changed after the review.
    summary=[f"Candidates: {n}", f"Original usable: {table["original_usable"].sum()}",
             f"Retained original: {(table["usable"] & table["original_usable"]).sum()}",
             f"Recovered: {table["recovered"].sum()}",f"Newly excluded: {table["newly_excluded"].sum()}",
             f"Final usable: {table["usable"].sum()}",
             f"Top 100 by peak, usable: {table["original_usable"][top].sum()} -> {table["usable"][top].sum()}",
             "Reasons among newly excluded (can overlap): " + str(Counter(s for f in table["flags"][table["newly_excluded"]] for s in f.split("|"))),
             "Original reasons among recovered: " + str(Counter(map(str,table["original_flags"][table["recovered"]]))),
             f"Input SHA256: {input_hash}", "Settings: " + table.meta["limits"]]
    (OUTPUT / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary),flush=True)

    save_plots(table, signal, rms, bad)

    return table


if __name__=="__main__":
    run()
