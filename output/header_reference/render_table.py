from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
rows = [
('OBSERVATION AND IMAGE', '', ''),
('OBJECT', 'Lockman_mosaic_r_1046+5902…', 'Target / field identifier (abbreviated).'),
('TELESCOP', 'KPNO 4.0 meter telescope', 'Telescope used for the observation.'),
('FILTER', 'r SDSS 1018', 'Filter identifier; SDSS r-band filter, not proof of SDSS survey origin.'),
('DATE-OBS', '2002-02-19T06:12:10.7', 'Observation start time; approximately UTC according to TIMESYS.'),
('EXPTIME', '720 s', 'Recorded exposure time; verify applicability to the final mosaic.'),
('AIRMASS', '1.207', 'Relative atmospheric path length; approximately 1 at the zenith.'),
('RA / DEC', '10:46:00.00 / +59:01:59.99', 'Recorded pointing: right ascension (h:m:s) and declination (deg:m:s).'),
('NAXIS', '2', 'Number of data axes: a two-dimensional image.'),
('NAXIS1 / NAXIS2', '2570 / 4611 pixels', 'Width / height. NumPy data.shape is (4611, 2570).'),
('PIXSCAL1 / PIXSCAL2', '0.258 / 0.258 arcsec/pixel', 'Recorded angular pixel scale; verify against the final image WCS.'),
('DETECTOR / NCCDS', 'CCDMosaThin1 / 8', 'Original detector identifier / number of CCDs; not the number of HDUs here.'),
('SIGNAL AND PHOTOMETRY', '', ''),
('GAIN', '3.1 electrons/ADU', 'Recorded electrons per digital count; verify the effective gain after processing.'),
('RDNOISE', '5.6 electrons', 'Recorded CCD readout noise; useful for uncertainty estimates.'),
('SATURATE', '36000 ADU', 'Header-labelled maximum good data value; verify for this processed image.'),
('MAGZPT', '25.30 mag', 'Photometric zero point for default extinction. Check the pipeline convention before use.'),
('MAGZRR', '0.02 mag', 'Reported uncertainty in the photometric zero point.'),
('PERCORR', '-0.006 mag', 'Header-labelled sky calibration correction; application is pipeline-specific.'),
('BUNIT', 'Not present', 'Physical unit is not explicitly declared. ADU is inferred from GAIN / SATURATE comments.'),
('SKY COORDINATES (WCS)', '', ''),
('CTYPE1 / CTYPE2', 'RA---ZPN / DEC--ZPN', 'Celestial axes and projection: right ascension / declination, ZPN projection.'),
('CRPIX1 / CRPIX2', '4493.0 / 4219.0', 'Reference pixel coordinates, using FITS 1-based indexing; may lie outside the image.'),
('CRVAL1 / CRVAL2', '161.52225 / 59.028858 deg', 'Celestial coordinates at the WCS reference pixel.'),
('CD1_1, CD1_2,\nCD2_1, CD2_2', '2 × 2 matrix in header', 'Linear mapping from pixel offsets to projection-plane coordinates (deg/pixel).'),
('PV2_1 / PV2_3', '1 / 375', 'Projection parameters used with the ZPN WCS.'),
('RADECSYS / EQUINOX', 'FK5 / 2000.0', 'Celestial reference system / reference equinox.'),
('STDCRMS / NUMBRMS', '0.240 arcsec / 53', 'Reported astrometric fit error / number of standards used.'),
('FITS STORAGE', '', ''),
('BITPIX', '16', 'Pixels are stored as signed 16-bit integers on disk.'),
('BSCALE / BZERO', '1 / 32768', 'Read value = stored value × BSCALE + BZERO. Astropy applies this automatically.'),
('NEXTEND', '8 (inconsistent)', 'Legacy extension count; hdul.info() shows only one HDU in this file.'),
]
wrapped = [[textwrap.fill(x, w, break_long_words=False) if '\n' not in x else x for x,w in zip(row,[24,34,77])] for row in rows]
heights = [0.35 if not r[1] else max(0.48, 0.19 * max(x.count('\n')+1 for x in wr)+0.18) for r,wr in zip(rows,wrapped)]
fig_h = sum(heights)+2.4
fig, ax = plt.subplots(figsize=(15,fig_h))
fig.patch.set_facecolor('#ffffff'); ax.axis('off')
fig.subplots_adjust(left=.025,right=.975,top=1-1.05/fig_h,bottom=.95/fig_h)
table=ax.table(cellText=wrapped,colLabels=['HEADER KEYWORD','VALUE IN THIS FILE','MEANING / USE'],colWidths=[.20,.28,.52],cellLoc='left',loc='upper center',bbox=[0,0,1,1])
table.auto_set_font_size(False);table.set_fontsize(10)
total=sum(heights)+.48
for (i,j),cell in table.get_celld().items():
    cell.set_edgecolor('#dbe3ed');cell.set_linewidth(.5);cell.PAD=.045
    if i==0:
        cell.set_facecolor('#17334f');cell.set_text_props(color='white',weight='bold');cell.set_height(.48/total)
    else:
        cell.set_height(heights[i-1]/total)
        if not rows[i-1][1]:
            cell.set_facecolor('#dce9f5');cell.set_text_props(weight='bold',color='#17334f')
        else:cell.set_facecolor('#f5f8fc' if i%2 else 'white')
fig.text(.025,1-.25/fig_h,'mosaic.fits | Important FITS Header Keywords',fontsize=23,weight='bold',color='#17334f',va='top')
fig.text(.025,1-.66/fig_h,'Values read from the local file • English reference table',fontsize=11,color='#52657b',va='top')
fig.text(.025,.69/fig_h,'Caution: this mosaic retains legacy detector metadata. Verify gain, exposure, saturation and WCS before quantitative use.',fontsize=10,color='#52657b')
fig.text(.025,.41/fig_h,'Sources: local FITS header; NASA FITS keyword dictionaries; NOAO/IRAF documentation. No survey identity is assumed.',fontsize=9,color='#52657b')
fig.savefig(out/'fits_header_reference_en.png',dpi=180,facecolor='white')
print(out/'fits_header_reference_en.png')
