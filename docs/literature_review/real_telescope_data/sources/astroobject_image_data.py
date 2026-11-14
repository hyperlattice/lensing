__author__ = 'sibirrer'


#external modules
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.coordinates as coords
import numpy as np

import astrofunc.util as util
import pyextract.image_config as ImageConfig

#internal modules
from astroObjectAnalyser.DataAnalysis.analysis import Analysis
from astroObjectAnalyser.DataAnalysis.catalogues import Catalogue


class StrongLensImageData(object):
    """
    contains all the information associated with a given band image (e.g.  r-band frame of specific lens)
    """

    def __init__(self, local_filename=None, local_psf_filename=None, local_wht_filename=None, ra=None, dec=None,
                 ra_cutout_cent=None, dec_cutout_cent=None, cutout_scale=None, data_type='cosmos', wht_extension=1,
                 sci_extension=0):

        """
        initialize data class with file names (absolute paths), coordinates and data type specifications
        """
        self.ra = ra
        self.dec = dec
        self.ra_cutout_cent = ra_cutout_cent
        self.dec_cutout_cent = dec_cutout_cent
        self.cutout_scale = cutout_scale

        self.local_filename = local_filename
        self.local_psf_filename = local_psf_filename
        self.local_wht_filename = local_wht_filename

        self.catalogue = Catalogue()
        self.analysis = Analysis()
        self.data_type = data_type
        self._extension_image = sci_extension  # or'SCI'
        self._extension_wht = wht_extension  # or 'WHT'

    @property
    def data_cutout(self):
        if not hasattr(self, '_data_cutout'):
            self.image_cutout(self.ra_cutout_cent, self.dec_cutout_cent, self.cutout_scale)
        return self._data_cutout

    def del_cutout(self):
        if hasattr(self, '_data_cutout'):
            del self._data_cutout
            del self._ra_coords_cutout
            del self._dec_coords_cutout
            del self._header_cutout

    @property
    def header_cutout(self):
        if not hasattr(self, '_header_cutout'):
            print('WARNING: New cutout image is built with default number of pixels.')
            self.image_cutout(self.ra_cutout_cent, self.dec_cutout_cent, self.cutout_scale)
        return self._header_cutout

    @property
    def header_primary(self):
        if not hasattr(self, '_header_primary'):
            self.header_info()
        return self._header_primary

    @property
    def header(self):
        if not hasattr(self, '_header'):
            self.header_info()
        return self._header

    @property
    def cd1(self):
        if not hasattr(self, '_cd1'):
            self.pixel_scale()
        return self._cd1

    @property
    def cd2(self):
        if not hasattr(self, '_cd2'):
            self.pixel_scale()
        return self._cd2

    @property
    def naxis1(self):
        if not hasattr(self, '_naxis1'):
            self.pixel_number()
        return self._naxis1

    @property
    def naxis2(self):
        if not hasattr(self, '_naxis2'):
            self.pixel_number()
        return self._naxis2

    @property
    def exposure_time(self):
        if not hasattr(self, '_exposure_time'):
            self._exposure_time = self.header_primary.get('EXPTIME')
        return self._exposure_time

    @property
    def exposure_map(self):
        if not hasattr(self, '_exposure_map'):
            self.image_cutout(self.ra_cutout_cent, self.dec_cutout_cent, self.cutout_scale, exposure_map=True)
        return self._exposure_map

    @property
    def CCD_gain(self):
        if not hasattr(self, '_CCD_gain'):
            if self.data_type in ['DES', 'GEMINI']:
                self._CCD_gain = self.header.get('GAIN')
            else:
                self._CCD_gain = self.header_primary.get('CCDGAIN')
            print("ccd gain =", self._CCD_gain)
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

    def get_psf_from_file(self, kernelsize):
        if not hasattr(self, '_psf_data'):
            self._get_psf_from_file()
        kernel = util.cut_psf(self._psf_data, kernelsize)
        return kernel

    def psf_fit(self, psf_type):
        private = '_' + psf_type
        if not hasattr(self, private):
            _, psf_variables = self._get_psf_fit(psf_type)
            setattr(self, private, psf_variables)
        return getattr(self, private)

    def psf_kernel(self, kernelsize):
        if not hasattr(self, '_psf_kernel'):
            kernel, psf_variables = self._get_psf_fit(psf_type='moffat')
            self._psf_kernel = kernel
            if not hasattr(self, '_moffat'):
                self._moffat = psf_variables
        kernel = util.cut_psf(self._psf_kernel, kernelsize)
        return kernel

    @property
    def get_cat(self):
        if not hasattr(self, '_cat'):
            cat = self._get_cat()
            self._cat = cat
        return self._cat

    @property
    def get_cutout_coords(self):
        if not hasattr(self, '_ra_coords_cutout') or not hasattr(self, '_dec_coords_cutout'):
            self.image_cutout(self.ra_cutout_cent, self.dec_cutout_cent, self.cutout_scale)
        return self._ra_coords_cutout, self._dec_coords_cutout

    @property
    def transforms(self):
        if not hasattr(self, '_pix2coord_transform') or not hasattr(self, '_coord2pix_transform'):
            self._transform()
        return self._pix2coord_transform, self._coord2pix_transform

    @property
    def transforms_undistorted(self):
        if not hasattr(self, '_pix2coord_transform_undistorted') or not hasattr(self, '_coord2pix_transform_undistorted'):
            self._transform_undistorted()
        return self._pix2coord_transform_undistorted, self._coord2pix_transform_undistorted

    def get_subgrid(self, subgrid_res=2):
        if not hasattr(self, '_ra_coords_cutout') or not hasattr(self, '_dec_coords_cutout'):
            self.image_cutout(self.ra_cutout_cent, self.dec_cutout_cent, self.cutout_scale)
        cos_dec = np.cos(self.dec / 360 * 2 * np.pi)
        ra_coords_sub, dec_coord_sub = util.make_subgrid(self._ra_coords_cutout*cos_dec, self._dec_coords_cutout, subgrid_res)

        return ra_coords_sub, dec_coord_sub

    def _get_background(self):
        """

        :return: mean and rms value of background
        """
        HDUFile = self.get_HDUFile()
        mean, rms = self.catalogue.get_background(HDUFile)
        return mean, rms

    def _get_cat(self):
        """

        :return: sextractor catalogue
        """
        HDUFile = self.get_HDUFile()
        cat = self.catalogue.get_source_cat(HDUFile)
        return cat

    def _get_psf_fit(self, psf_type):
        """

        :param psf_type:
        :return:
        """
        exp_time = self.exposure_time
        HDUFile = self.get_HDUFile()
        mean, rms = self.catalogue.get_background(HDUFile)
        cat = self.catalogue.get_source_cat(HDUFile)
        image = self.image_full()
        kernel, mean_list, filter_object = self.analysis.get_psf(image, cat, mean, rms, exp_time, psf_type)
        return kernel, mean_list

    def get_HDUFile(self, force=False):
        if not(hasattr(self, '_HDUFile') and hasattr(self, '_image_no_border') and (not force)):
            #image = self.image_full()
            image_path = self.local_filename
            exp_time = self.exposure_time
            CCD_gain = self.CCD_gain

            conf_args = ImageConfig.config_arguments(exp_time, CCD_gain)
            self._HDUFile = ImageConfig.get_source_cat(imageref=image_path, conf_args=conf_args)
        return self._HDUFile

    def _get_psf_from_file(self):
        """
        loads in psf data stored in Tiny_Tim folder of the lens system
        :return:
        """
        self._psf_data = pyfits.getdata(self.local_psf_filename)

    def header_info(self):
        """
        reads in the header info and performs checks on whether the header has the right format to deal with
        """
        self._header_primary = pyfits.getheader(self.local_filename) # this is the primary header which does not contain general information
        file = pyfits.open(self.local_filename)
        self._header = file[self._extension_image].header
        #else:
        #    self._header = file['SCI'].header  # this is the header of the science image
        file.close()

    def pixel_scale(self):
        """
        returns the pixel scale of the image (units still unclear!!!)
        """
        if not hasattr(self, 'header'):
            self.header_info()
        self._cd1 = self.header.get('CDELT1') if self.header.get('CDELT1') else np.sqrt(self.header.get('CD1_1')**2 + self.header.get('CD1_2')**2)
        self._cd2 = self.header.get('CDELT2') if self.header.get('CDELT2') else np.sqrt(self.header.get('CD2_1')**2 + self.header.get('CD2_2')**2)
        if self.cd1 is None or self.cd2 is None:
            raise Exception("Missing CD or CDELT keywords in header")

    def pixel_number(self):
        """
        reads in number of pixel per axis for original image
        """
        if self.header['NAXIS'] > 2:
            raise TypeError("Too many (%i) dimensions!" % self.header['NAXIS'])
        self._naxis1 = self.header['NAXIS1']
        self._naxis2 = self.header['NAXIS2']

    def image_full(self):
        """
        array of one full band, do only use this function when really needed as images can be quite large
        """
        file = pyfits.open(self.local_filename)
        data_full = file[self._extension_image].data
        data_full[np.isnan(data_full)] = 0
        file.close()
        return data_full

    def exposure_full(self):
        """
        array of one full band exposure time. do only use this function when really needed as images can be quite large
        """
        if self.local_wht_filename is not None:
            file = pyfits.open(self.local_wht_filename)
            exp_full = file[0].data
            print("separate exposure map loaded")
        else:
            file = pyfits.open(self.local_filename)
            exp_full = file[self._extension_wht].data
            #else:
            #    exp_full = file['WHT'].data
        exp_full[np.isnan(exp_full)] = 0
        file.close()
        return exp_full

    def image_cutout(self, xc, yc, cutout_scale, cutout_filename=None, exposure_map=False):
        """
        used to make a cut out image. Can also be called after the instance has been initialised, in which case
        the cutout image will be updated (i.e. changed).
        """
        # if not hasattr(self,'header'):
        #     self.header_info()
        if isinstance(cutout_filename, str):
            file = pyfits.open(cutout_filename)
            if self.data_type == 'cosmos' or self.data_type == 'DES':
                self._header_cutout = file[0].header
                self._data_cutout = file[0].data
            elif self.data_type == 'GEMINI':
                self._header_cutout = file[self._extension_image].header
                self._header_cutout = file[self._extension_image].data
            else:
                self._header_cutout = file['SCI'].header
                self._data_cutout = file['SCI'].data
            file.close()
            self._xmin_c = 0
            self._ymin_c = 0
        else:
            if cutout_scale is None:
                cutout_scale = 50
                print("New cutout is generated with default cutout scale.")
            xw, yw = int(cutout_scale/2), int(cutout_scale/2)
            self._data_cutout, self._header_cutout, exp_map, self._ra_coords_cutout, self._dec_coords_cutout =\
                self._cutout(self.local_filename, xc, yc, xw, yw, exposure_map=exposure_map)
            if exposure_map is True:
                self._exposure_map = exp_map
        pass

    def _cutout(self, fits_filename, xc, yc, xw, yw, units='pixels',
            clobber=True, useMontage=False, coordsys='galactic', verbose=False, exposure_map=False):
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


        head = self.header.copy()
        wcs = pywcs.WCS(head)
        if units == 'wcs':
            if coordsys == 'celestial' and wcs.wcs.lngtyp == 'GLON':
                xc, yc = coords.Position((xc, yc), system=coordsys).galactic()
            elif coordsys == 'galactic' and wcs.wcs.lngtyp == 'RA':
                xc, yc = coords.Position((xc, yc), system=coordsys).j2000()
        xx, yy = wcs.all_world2pix(xc, yc, 0)
        print('the center of the image is at pixel coordinates %f, %f.' % (xx, yy))
        if units == 'pixels':
            xmin, xmax = np.max([0, xx-xw]), np.min([self.naxis1, xx+xw])
            ymin, ymax = np.max([0, yy-yw]), np.min([self.naxis2, yy+yw])
        elif units == 'wcs':
            xmin, xmax = np.max([0, xx-xw/np.abs(self.cd1)]), np.min([self.naxis1, xx+xw/np.abs(self.cd1)])
            ymin, ymax = np.max([0, yy-yw/np.abs(self.cd2)]), np.min([self.naxis2, yy+yw/np.abs(self.cd2)])
        else:
            raise Exception("Can't use units %s." % units)
        self._xmin_c, self._xmax_c = xmin, xmax
        self._ymin_c, self._ymax_c = ymin, ymax
        if xmax < 0 or ymax < 0:
            raise ValueError("Max Coordinate is outside of map: %f,%f." % (xmax,ymax))
        if ymin >= head.get('NAXIS2') or xmin >= head.get('NAXIS1'):
            raise ValueError("Min Coordinate is outside of map: %f,%f." % (xmin,ymin))

        head = self.change_header(head, xmin, xmax, ymin, ymax)
        img = self.image_full()[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
        # img = file['SCI'].data[ymin:ymax, xmin:xmax]
        if verbose: print("Cut image %s to %s. xrange: %f:%f, yrange: %f:%f" % (fits_filename, img.shape, xmin, xmax, ymin, ymax))
        if exposure_map:
            exp_map = self.exposure_full()[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
        else:
            exp_map = None
        ra_coord, dec_coord = self.get_coordinates(xmin, xmax, ymin, ymax, wcs)
        return img, head, exp_map, ra_coord, dec_coord

    def change_header(self, head, xmin, xmax, ymin, ymax):
        """
        changes the header to adjust information to the cutout image.
        Attention, this does not mitigate distortion correction terms!
        """
        head['CRPIX1'] -= xmin
        head['CRPIX2'] -= ymin
        head['NAXIS1'] = int(xmax-xmin)
        head['NAXIS2'] = int(ymax-ymin)
        if head.get('NAXIS1') == 0 or head.get('NAXIS2') == 0:
            raise ValueError("Map has a 0 dimension: %i,%i." % (head.get('NAXIS1'), head.get('NAXIS2')))
        return head

    def get_coordinates(self, x_min, x_max, y_min, y_max, wcs):
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
        x_coords = np.linspace(int(x_min), int(x_max - 1), int(x_max - x_min))
        y_coords = np.linspace(int(y_min), int(y_max - 1), int(y_max - y_min))
        x_coords, y_coords = np.meshgrid(x_coords, y_coords)
        ra_coords, dec_coords = wcs.all_pix2world(x_coords, y_coords, 0)
        ra_coords -= self.ra
        dec_coords -= self.dec
        ra_coords *= 3600
        dec_coords *= 3600
        ra_coords = util.image2array(ra_coords)
        dec_coords = util.image2array(dec_coords)
        return ra_coords, dec_coords

    @property
    def pixel_at_angle_0(self):
        """

        :return: pixel coordinate (x_0, y_0) of (ra,dec) = (0,0) for cutout image
        """
        head = self.header
        wcs = pywcs.WCS(head)
        x_0, y_0 = wcs.all_world2pix(self.ra, self.dec, 0)
        return x_0 - self._xmin_c, y_0 - self._ymin_c

    @property
    def coord_at_pixel_0(self):
        """

        :return: angular coordinate (relative arc sec) (ra_0, dec_0) of (pix_x,pix_y) = (0,0)
        """
        #head = self.header
        #wcs = pywcs.WCS(head)
        x0, y0 = self.pixel_at_angle_0
        _pix2coord_transform, _coord2pix_transform = self.transforms
        ra_pos, dec_pos = util.map_coord2pix(-x0, -y0, 0, 0, _pix2coord_transform)
        #ra_0, dec_0 = wcs.all_pix2world(self._xmin_c, self._ymin_c, 0)
        #cos_dec = np.cos(self.dec / 360 * 2 * np.pi)
        #d_ra = (ra_0 - self.ra) * 3600. * cos_dec
        #d_dec = (dec_0 - self.dec) * 3600.
        #return d_ra, d_dec
        return ra_pos, dec_pos

    def map_coord2pix(self, ra, dec):
        """

        :param ra: ra coordinates, relative
        :param dec: dec coordinates, relative
        :return: x, y pixel coordinates
        """
        x_0, y_0 = self.pixel_at_angle_0
        _pix2coord_transform, _coord2pix_transform = self.transforms
        x_pos, y_pos = util.map_coord2pix(ra, dec, x_0, y_0, _coord2pix_transform)
        return x_pos, y_pos

    def map_pix2coord(self, x_pos, y_pos):
        """

        :param x_pos: pixel coordinate
        :param y_pos: pixel coordinate
        :return: relative ra, dec coordinate
        """
        ra_0, dec_0 = self.coord_at_pixel_0
        _pix2coord_transform, _coord2pix_transform = self.transforms
        ra_pos, dec_pos = util.map_coord2pix(x_pos, y_pos, ra_0, dec_0, _pix2coord_transform)
        return ra_pos, dec_pos

    @property
    def numPix(self):
        return int(self.cutout_scale/2) * 2

    def _transform_undistorted(self, approx=False):
        """
        initializes the the matrix which transforms pixel to ra/dec
        """
        if not hasattr(self, 'header'):
            self.header_info()

        CD1_1 = self.header.get('CD1_1')*3600  # change in arc sec per pixel d(ra)/dx
        CD1_2 = self.header.get('CD1_2')*3600
        CD2_1 = self.header.get('CD2_1')*3600
        CD2_2 = self.header.get('CD2_2')*3600

        self._pix2coord_transform_undistorted = np.array([[CD1_1, CD1_2], [CD2_1, CD2_2]])
        det = CD1_1*CD2_2 - CD1_2*CD2_1
        self._coord2pix_transform_undistorted = np.array([[CD2_2, -CD1_2], [-CD2_1, CD1_1]])/det

    def _transform(self):
        """

        :return: linear transformation matrix for the pixel shift at (0,0) comupted with the full distortion corrections
        """
        head = self.header
        wcs = pywcs.WCS(head)
        xc, yc = self.pixel_at_angle_0
        ra_0, dec_0 = wcs.all_pix2world(xc, yc, 0)
        ra_10, dec_10 = wcs.all_pix2world(xc+1, yc, 0)
        ra_01, dec_01 = wcs.all_pix2world(xc, yc+1, 0)
        cos_dec = np.cos(self.dec / 360 * 2 * np.pi)
        factor = 3600.
        CD1_1 = (ra_10 - ra_0) * factor * cos_dec
        CD1_2 = (dec_10 - dec_0) * factor
        CD2_1 = (ra_01 - ra_0) * factor * cos_dec
        CD2_2 = (dec_01 - dec_0) * factor

        self._pix2coord_transform = np.array([[CD1_1, CD1_2], [CD2_1, CD2_2]])
        det = CD1_1*CD2_2 - CD1_2*CD2_1
        self._coord2pix_transform = np.array([[CD2_2, -CD1_2], [-CD2_1, CD1_1]])/det

    def _transform_large(self):
        """

        :return: linear transformation matrix for the pixel shift at (0,0) comupted with the full distortion corrections
        """
        head = self.header
        wcs = pywcs.WCS(head)
        xc, yc = self.pixel_at_angle_0
        ra_0, dec_0 = wcs.all_pix2world(xc, yc, 0)
        ra_10, dec_10 = wcs.all_pix2world(xc+100, yc, 0)
        ra_01, dec_01 = wcs.all_pix2world(xc, yc+100, 0)
        cos_dec = np.cos(self.dec / 360 * 2 * np.pi)
        factor = 3600. * 100
        CD1_1 = (ra_10 - ra_0) * factor * cos_dec
        CD1_2 = (dec_10 - dec_0) * factor
        CD2_1 = (ra_01 - ra_0) * factor * cos_dec
        CD2_2 = (dec_01 - dec_0) * factor

        self._pix2coord_transform = np.array([[CD1_1, CD1_2], [CD2_1, CD2_2]])
        det = CD1_1*CD2_2 - CD1_2*CD2_1
        self._coord2pix_transform = np.array([[CD2_2, -CD1_2], [-CD2_1, CD1_1]])/det
