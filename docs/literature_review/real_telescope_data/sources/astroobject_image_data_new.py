__author__ = 'sibirrer'


#external modules
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import numpy as np
import pyextract.image_config as ImageConfig
import astropy.coordinates as coords

#internal modules
from astroObjectAnalyser.DataAnalysis.analysis import Analysis
from astroObjectAnalyser.DataAnalysis.catalogues import Catalogue


class ImageData(object):
    """
    contains all the information associated with a given band image (e.g.  r-band frame of specific lens)
    """

    def __init__(self, image_filename=None, wht_filename=None, data_type='cosmos', wht_extension=1, sci_extension=0):

        """
        initialize data class with file names (absolute paths), coordinates and data type specifications
        """
        self._image_filename = image_filename
        self._wht_filename = wht_filename

        self.catalogue = Catalogue()
        self.analysis = Analysis()
        self._data_type = data_type
        self._extension_image = sci_extension  # or'SCI'
        self._extension_wht = wht_extension  # or 'WHT'

    @property
    def header_primary(self):
        if not hasattr(self, '_header_primary'):
            self._load_header()
        return self._header_primary

    @property
    def header(self):
        if not hasattr(self, '_header'):
            self._load_header()
        return self._header

    @property
    def naxis1(self):
        if not hasattr(self, '_naxis1'):
            self._pixel_number()
        return self._naxis1

    @property
    def naxis2(self):
        if not hasattr(self, '_naxis2'):
            self._pixel_number()
        return self._naxis2

    @property
    def exposure_time(self):
        if not hasattr(self, '_exposure_time'):
            self._exposure_time = self.header_primary.get('EXPTIME')
        return self._exposure_time

    @property
    def CCD_gain(self):
        if not hasattr(self, '_CCD_gain'):
            if 'CCDGAIN' in self.header_primary:
                self._CCD_gain = self.header_primary.get('CCDGAIN')
            elif 'CCDGAIN' in self.header:
                self._CCD_gain = self.header.get('CCDGAIN')
            elif 'GAIN' in self.header:
                self._CCD_gain = self.header.get('GAIN')
            else:
                raise ValueError("CCD gain could not be read from the header. Please manually add it!")
        return self._CCD_gain

    @property
    def background(self):
        if not hasattr(self, '_background'):
            self._background_mean, self._background_rms = self._get_background()
        return self._background_mean, self._background_rms

    def set_extension(self, ext_image=0, ext_wht=1):
        """"""
        self._extension_image = ext_image
        self._extension_wht = ext_wht

    @property
    def get_cat(self):
        if not hasattr(self, '_cat'):
            cat = self._get_cat()
            self._cat = cat
        return self._cat

    def transforms(self, xc, yc):
        if not hasattr(self, '_pix2coord_transform') or not hasattr(self, '_coord2pix_transform'):
            self.transform(xc, yc)
        return self._pix2coord_transform, self._coord2pix_transform

    @property
    def transforms_undistorted(self):
        if not hasattr(self, '_pix2coord_transform_undistorted') or not hasattr(self, '_coord2pix_transform_undistorted'):
            self._transform_undistorted()
        return self._pix2coord_transform_undistorted, self._coord2pix_transform_undistorted

    def _get_background(self):
        """

        :return: mean and rms value of background
        """
        HDUFile = self.HDUFile()
        mean, rms = self.catalogue.get_background(HDUFile)
        return mean, rms

    def _get_cat(self):
        """

        :return: sextractor catalogue
        """
        HDUFile = self.HDUFile()
        cat = self.catalogue.get_source_cat(HDUFile)
        return cat

    def _get_psf_fit(self, psf_type):
        """

        :param psf_type:
        :return:
        """
        exp_time = self.exposure_time
        HDUFile = self.HDUFile()
        mean, rms = self.catalogue.get_background(HDUFile)
        cat = self.catalogue.get_source_cat(HDUFile)
        image = self.image_full()
        kernel, mean_list, filter_object = self.analysis.get_psf(image, cat, mean, rms, exp_time, psf_type)
        return kernel, mean_list

    def HDUFile(self, force=False):
        if not(hasattr(self, '_HDUFile') and (not force)):
            conf_args = ImageConfig.config_arguments(self.exposure_time, self.CCD_gain)
            self._HDUFile = ImageConfig.get_source_cat(imageref=self._image_filename, conf_args=conf_args)
        return self._HDUFile

    def _load_header(self):
        """
        reads in the header info and performs checks on whether the header has the right format to deal with
        """
        self._header_primary = pyfits.getheader(self._image_filename) # this is the primary header which does not contain general information
        file = pyfits.open(self._image_filename)
        self._header = file[self._extension_image].header
        file.close()

    def _pixel_number(self):
        """
        reads in number of pixel per axis for original image
        """
        if self.header['NAXIS'] > 2:
            raise TypeError("Too many (%i) dimensions!" % self.header['NAXIS'])
        self._naxis1 = self.header['NAXIS1']
        self._naxis2 = self.header['NAXIS2']

    @property
    def image_full(self):
        """
        array of one full band, do only use this function when really needed as images can be quite large
        """
        file = pyfits.open(self._image_filename)
        data_full = file[self._extension_image].data
        data_full[np.isnan(data_full)] = 0
        file.close()
        return data_full

    @property
    def exposure_full(self):
        """
        array of one full band exposure time. do only use this function when really needed as images can be quite large
        """
        if self._wht_filename is not None:
            file = pyfits.open(self._wht_filename)
            exp_full = file[0].data
            print("separate exposure map loaded")
        else:
            file = pyfits.open(self._image_filename)
            exp_full = file[self._extension_wht].data
            #else:
            #    exp_full = file['WHT'].data
        exp_full[np.isnan(exp_full)] = 0
        file.close()
        return exp_full

    @property
    def pixel_size(self):
        """

        :return: pixel size in arc seconds
        """
        cd1 = self.header.get('CDELT1') if self.header.get('CDELT1') else np.sqrt(
            self.header.get('CD1_1') ** 2 + self.header.get('CD1_2') ** 2)
        cd2 = self.header.get('CDELT2') if self.header.get('CDELT2') else np.sqrt(
            self.header.get('CD2_1') ** 2 + self.header.get('CD2_2') ** 2)
        if cd1 is None or cd2 is None:
            raise Exception("Missing CD or CDELT keywords in header")
        return cd1 * 3600, cd2 * 3600

    def coordinates_grid(self, x_min, x_max, y_min, y_max):
        """
        :param xc: center in ra
        :param yc: center in dec
        :param x_min: min pixel x-axis
        :param x_max: max pixel x-axis
        :param y_min: min pixel y-axis
        :param y_max: max pixel y-axis
        :param wcs: coordinate class initialized with the fits file of the original image
        :return: ra_coord, dec_coord in units of arc seconds centered to the cutout position
        """
        head = self.header
        wcs = pywcs.WCS(head)
        x_coords = np.linspace(x_min, x_max-1, x_max - x_min)
        y_coords = np.linspace(y_min, y_max-1, y_max - y_min)
        x_coords, y_coords = np.meshgrid(x_coords, y_coords)
        ra_coords, dec_coords = wcs.all_pix2world(x_coords, y_coords, 0)
        #ra_coords -= self.ra
        #dec_coords -= self.dec
        #ra_coords *= 3600
        #dec_coords *= 3600
        #ra_coords = util.image2array(ra_coords)
        #dec_coords = util.image2array(dec_coords)
        return ra_coords, dec_coords

    def pix2coord(self, x, y):
        """
        maps pixel indices to ra/dec coordinates

        :param x:
        :param y:
        :return:
        """
        wcs = pywcs.WCS(self.header)
        return wcs.all_pix2world(x, y, 0)

    def coord2pix(self, ra, dec):
        """
        maps ra/dec coordinates to pixel indices

        :param x:
        :param y:
        :return:
        """
        wcs = pywcs.WCS(self.header)
        return wcs.all_world2pix(ra, dec, 0)

    def cutout_range(self, rac, decc, xw, yw, units='pixels', coordsys='galactic'):
        """
        computes the pixel range (min max) of a frame centered at (rac, decc) with width xw, yw (in arc seconds)

        :param rac:
        :param decc:
        :param xw:
        :param yw:
        :return:
        """
        head = self.header
        wcs = pywcs.WCS(head)
        if units == 'wcs':
            if coordsys == 'celestial' and wcs.wcs.lngtyp == 'GLON':
                rac, decc = coords.Position((rac, decc), system=coordsys).galactic()
            elif coordsys == 'galactic' and wcs.wcs.lngtyp == 'RA':
                rac, decc = coords.Position((rac, decc), system=coordsys).j2000()
            else:
                raise ValueError("problem with wcs instance.")
        xx, yy = wcs.all_world2pix(rac, decc, 0)
        xx = int(xx)
        yy = int(yy)
        print('the center of the image is at pixel coordinates %f, %f.' % (xx, yy))
        if units == 'pixels':
            xmin, xmax = np.max([0, xx - xw]), np.min([self.naxis1, xx + xw])
            ymin, ymax = np.max([0, yy - yw]), np.min([self.naxis2, yy + yw])
        elif units == 'arcseconds':
            cd1, cd2 = self.pixel_size
            xmin, xmax = np.max([0, xx - xw / np.abs(cd1)]), np.min([self.naxis1, xx + xw / np.abs(cd1)])
            ymin, ymax = np.max([0, yy - yw / np.abs(cd2)]), np.min([self.naxis2, yy + yw / np.abs(cd2)])
        else:
            raise Exception("Can't use units %s." % units)
        if xmax < 0 or ymax < 0:
            raise ValueError("Max Coordinate is outside of map: %f,%f." % (xmax, ymax))
        if ymin >= head.get('NAXIS2') or xmin >= head.get('NAXIS1'):
            raise ValueError("Min Coordinate is outside of map: %f,%f." % (xmin, ymin))
        return xmin, xmax, ymin, ymax

    def cutout(self, xc, yc, xw, yw, units='pixels', coordsys='galactic', verbose=False,
               exposure_map=False):
        """
        Inputs:
            file  - pyfits HDUList (must be 2D)
            xc,yc - x and y coordinates in the fits files' coordinate system (CTYPE)
            xw,yw - x and y width (pixels or wcs)
            units - specify units to use: either pixels or wcs
            outfile - optional output file
        """
        # file = pyfits.open(fits_filename)
        # head = file['SCI'].header.copy()

        xmin, xmax, ymin, ymax = self.cutout_range(xc, yc, xw, yw, units=units, coordsys=coordsys)
        img = self.image_full[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
        if verbose is True: print("Cut image to %s. xrange: %f:%f, yrange: %f:%f" % (img.shape, xmin, xmax, ymin, ymax))
        if exposure_map is True:
            exp_map = self.exposure_full[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
        else:
            exp_map = None
        return img, exp_map

    def _transform_undistorted(self):
        """
        initializes the the matrix which transforms pixel to ra/dec
        """
        if not hasattr(self, 'header'):
            self._load_header()

        CD1_1 = self.header.get('CD1_1')*3600  # change in arc sec per pixel d(ra)/dx
        CD1_2 = self.header.get('CD1_2')*3600
        CD2_1 = self.header.get('CD2_1')*3600
        CD2_2 = self.header.get('CD2_2')*3600

        pix2coord_transform = np.array([[CD1_1, CD1_2], [CD2_1, CD2_2]])
        det = CD1_1*CD2_2 - CD1_2*CD2_1
        coord2pix_transform = np.array([[CD2_2, -CD1_2], [-CD2_1, CD1_1]])/det
        return pix2coord_transform, coord2pix_transform

    def transform(self, xc, yc):
        """
        :param xc: pixel of the center
        :param yc: pixel of the center
        :return: linear transformation matrix for the pixel shift at the center of the image
        comupted with the full distortion corrections
        """
        head = self.header
        wcs = pywcs.WCS(head)

        ra_0, dec_0 = wcs.all_pix2world(xc, yc, 0)
        ra_10, dec_10 = wcs.all_pix2world(xc+1, yc, 0)
        ra_01, dec_01 = wcs.all_pix2world(xc, yc+1, 0)
        cos_dec = np.cos(dec_0 / 360 * 2 * np.pi)
        factor = 3600.
        CD1_1 = (ra_10 - ra_0) * factor * cos_dec
        CD1_2 = (dec_10 - dec_0) * factor
        CD2_1 = (ra_01 - ra_0) * factor * cos_dec
        CD2_2 = (dec_01 - dec_0) * factor

        pix2coord_transform = np.array([[CD1_1, CD1_2], [CD2_1, CD2_2]])
        det = CD1_1*CD2_2 - CD1_2*CD2_1
        coord2pix_transform = np.array([[CD2_2, -CD1_2], [-CD2_1, CD1_1]])/det
        return pix2coord_transform, coord2pix_transform

    def _transform_large(self, xc, yc, delta_pix=100):
        """
        :param xc: pixel of the center
        :param yc: pixel of the center
        :return: linear transformation matrix for the pixel shift at (0,0) comupted with the full distortion corrections
        """
        head = self.header
        wcs = pywcs.WCS(head)
        ra_0, dec_0 = wcs.all_pix2world(xc, yc, 0)
        ra_10, dec_10 = wcs.all_pix2world(xc + delta_pix, yc, 0)
        ra_01, dec_01 = wcs.all_pix2world(xc, yc + delta_pix, 0)
        cos_dec = np.cos(dec_0 / 360 * 2 * np.pi)
        factor = 3600. / delta_pix
        CD1_1 = (ra_10 - ra_0) * factor * cos_dec
        CD1_2 = (dec_10 - dec_0) * factor
        CD2_1 = (ra_01 - ra_0) * factor * cos_dec
        CD2_2 = (dec_01 - dec_0) * factor

        pix2coord_transform = np.array([[CD1_1, CD1_2], [CD2_1, CD2_2]])
        det = CD1_1*CD2_2 - CD1_2*CD2_1
        coord2pix_transform = np.array([[CD2_2, -CD1_2], [-CD2_1, CD1_1]])/det
        return pix2coord_transform, coord2pix_transform
