import xarray as xr
# from xarray import DataArray, concat
import numpy as np
import pygix  # type: ignore
import fabio # fabio package for .edf imports
import pathlib
from typing import Union, Tuple
import pyFAI
# from PyHyperScattering.IntegrationUtils import DrawMask
# from tqdm.auto import tqdm 

class WAXSTransform:
    """ Description: Class for transforming GIWAXS data into different formats. 
    1. Use pygix to apply the missing wedge Ewald's sphere correction & convert to q-space
    2. Generate 2D plots of Qz vs Qxy corrected detector images
    3. Generate 2d plots of Q vs Chi images, with the option to apply the sin(chi) correction
    4. etc. 
    """
    
    def __init__(self, 
                 poniPath: Union[str, pathlib.Path], 
                 maskPath: Union[str, pathlib.Path, np.ndarray], 
                 inplane_config: str = 'q_xy', 
                 energy: float = 12.7,
                 incident_angle = 0):
        
        """
        Attributes:
        poniPath (pathlib Path or str): Path to .poni file for converting to q-space 
                                        & applying missing wedge correction
        maskPath (pathlib Path or str or np.array): Path to the mask file to use 
                                for the conversion, or a numpy array
        inplane_config (str): The configuration of the inplane. Default is 'q_xy'.
        energy (optional, float): Set energy if default energy in poni file is invalid
        """

        self.poniPath = pathlib.Path(poniPath)
        try:
            self.maskPath = pathlib.Path(maskPath)
        except TypeError:
            self.maskPath = maskPath
            
        self.inplane_config = inplane_config
        if energy:
            self.energy = energy
            self.wavelength = np.round((4.1357e-15*2.99792458e8)/(energy*1000), 13)
        else:
            self.energy = None
            self.wavelength = None
        
        if incident_angle:
            self.incident_angle = incident_angle
        else:
            self.incident_angle = 0.3
        
        self.integrate1d_da = None

    # def load_mask(self, da):
    def load_mask(self):
        """Load the mask file based on its file type."""

        if isinstance(self.maskPath, np.ndarray):
            return self.maskPath

        try:
            # if self.maskPath.suffix == '.json':
            #     draw = DrawMask(da)  
            #     draw.load(self.maskPath)
            #     return draw.mask
            # elif self.maskPath.suffix == '.edf':
            if self.maskPath.suffix == '.edf':
                return fabio.open(self.maskPath).data
            else:
                raise ValueError(f"Unsupported file type: {self.maskPath.suffix}")
        except Exception as e:
            print(f"An error occurred while loading the mask file: {e}")

    def initialize_pg_transform(self, incident_angle):
        """
        Initialize the pygix Transform object with the provided incident angle and optional wavelength.

        Parameters:
        - incident_angle (float): The incident angle for the pygix Transform object.

        Returns:
        - pg (pygix.Transform): The initialized pygix Transform object.
        """
        pg = pygix.Transform()
        pg.load(str(self.poniPath))
        pg.sample_orientation = 3  # could be an optional parameter
        pg.incident_angle = incident_angle
        
        if self.wavelength:
            pg.wavelength = self.wavelength

        return pg

    def get_incident_angle(self, attributes, default=0.3):
        # Hardcoded list of aliases to look for the incident angle
        incident_angle_aliases = ['incident_angle', 
                                  'thpos', 
                                  'th', 
                                  'theta', 
                                  'incidence', 
                                  'inc_angle', 
                                  'angle_of_incidence',
                                  'incang',
                                  'incangle',
                                  'inc_angle']
                                  
        for alias in incident_angle_aliases:
            if alias in attributes:
                # Extract numerical part from the string, including the decimal point
                numerical_part = ''.join(c for c in attributes[alias] if c.isdigit() or c == '.')
                # Check if the extracted part is a valid float number
                try:
                    return float(numerical_part)
                except ValueError:
                    pass
        print(f"Warning: Incident angle not found. Defaulting to {default}. "
            f"Allowed aliases for incident angle: {', '.join(incident_angle_aliases)}")
        return default
    
    def pg_convert(self, da):
        """
        Converts raw GIWAXS detector image to q-space data. Returns two DataArrays, Qz vs Qxy & Q vs Chi
        
        Inputs: Raw GIWAXS DataArray
        Outputs: Cartesian & Polar DataArrays
        """
        
        # - Add method variable for list of correction parameters (solid angle, flat field, darks, etc.)

        # incident_angle = self.get_incident_angle(da.attrs, ['incident_angle', 'thpos'], default=0.3)
        # pg = self.initialize_pg_transform(incident_angle)

        # Initialize pygix transform object
        pg = pygix.Transform()
        pg.load(str(self.poniPath))
        pg.sample_orientation = 3 # could add this as optional method parameter in the handle
        
        # Method to extract incident angle from attributes using list of possible aliases.
        def get_incident_angle(attributes, aliases, default=0.3):
            for alias in aliases:
                if alias in attributes:
                    # Extract numerical part from the string, including the decimal point
                    numerical_part = ''.join(c for c in attributes[alias] if c.isdigit() or c == '.')
                    # Check if the extracted part is a valid float number
                    try:
                        return float(numerical_part)
                    except ValueError:
                        pass
            print(f"Warning: Incident angle not found. Defaulting to {default}. "
                f"Allowed aliases for incident angle: {', '.join(aliases)}")
            return default

        # List of aliases to look for the incident angle
        incident_angle_aliases = ['incident_angle', 
                                  'thpos', 
                                  'th', 
                                  'theta', 
                                  'incidence', 
                                  'inc_angle', 
                                  'angle_of_incidence'
                                  'incang',
                                  'incangle'
                                  'inc_angle']

        # Get the incident angle using the aliases
        self.incident_angle = get_incident_angle(da.attrs, incident_angle_aliases)
        pg.incident_angle = self.incident_angle

        # update to grab incident angle from keylist properly.
        # pg.incident_angle = float(da.incident_angle[2:])
        if self.wavelength:
            pg.wavelength = self.wavelength

        # Load mask
        # mask = self.load_mask(da)
        mask = self.load_mask()

        
        # Cartesian 2D plot transformation
        recip_data, qxy, qz = pg.transform_reciprocal(da.data,
                                                      method='bbox',
                                                      unit='A',
                                                      mask=mask,
                                                      correctSolidAngle=True)
        
        recip_da = xr.DataArray(data=recip_data,
                                dims=['q_z', self.inplane_config],
                                coords={
                                    'q_z': ('q_z', qz, {'units': '1/Å'}),
                                    self.inplane_config: (self.inplane_config, qxy, {'units': '1/Å'})
                                },
                                attrs=da.attrs)

        # Polar 2D plot transformation
        caked_data, qr, chi = pg.transform_image(da.data, 
                                                 process='polar',
                                                 method = 'bbox',
                                                #  method = 'splitpix',
                                                 unit='q_A^-1',
                                                 mask=mask,
                                                 correctSolidAngle=True)

        caked_da = xr.DataArray(data=caked_data,
                            dims=['chi', 'qr'],
                            coords={
                                'chi': ('chi', chi, {'units': '°'}),
                                'qr': ('qr', qr, {'units': '1/Å'})
                            },
                            attrs=da.attrs)
        caked_da.attrs['inplane_config'] = self.inplane_config

        # Preseve time dimension if it is in the dataarray, for stacking purposes
        if 'time' in da.coords:
            recip_da = recip_da.assign_coords({'time': float(da.time)})
            recip_da = recip_da.expand_dims(dim={'time': 1})
            caked_da = caked_da.assign_coords({'time': float(da.time)})
            caked_da = caked_da.expand_dims(dim={'time': 1})

        return recip_da, caked_da

    def pg_integrate1D(self, dataarray, 
                       npt = 1024, 
                       method = 'bbox', 
                       correctSolidAngle=True, 
                       polarization_factor = None,
                       unit = None):
        """
        Integrate 2D GIWAXS Images, Select Integration Method, Select Image Corrections
        
        Inputs: Raw GIWAXS DataArray, Selected Methods
        Outputs: 1D integrated DataArray

        pg.integrate_1d Args:
            data (ndarray): 2D array from detector (raw image).
            npt (int): Number of points in output data.
            correctSolidAngle (bool): Correct for solid angle of each pixel.
            mask : ndarray
                Masked pixel array (same size as image) with 1 for masked
                pixels and 0 for valid pixels.
            method : str
                Integration method. Can be "np", "cython", "bbox",
                "splitpix", "lut" or "lut_ocl" (if you want to go on GPU).
            unit : str
                Radial units. Can be "2th_deg", "2th_rad", "q_nm^-1" or
                "q_A^-1" (TTH_DEG, TTH_RAD, Q_NM, Q_A).
        """
        
        # Initialize pygix transform object
        # pg = pygix.Transform()
        # pg.load(str(self.poniPath))
        # pg.sample_orientation = 3  # could add this as an optional method parameter in the handle
        
        # Get & Set Incidence Angle using the get_incident_angle method
        self.incident_angle = self.get_incident_angle(dataarray.attrs)
        pg = self.initialize_pg_transform(self.incident_angle)
        pg.incident_angle = self.incident_angle

        # # Set Wavelength if available
        # if self.wavelength:
        #     pg.wavelength = self.wavelength

        # Load Mask
        # mask = self.load_mask(dataarray)
        mask = self.load_mask()

        if unit is not None:
            self.unit = unit
        else:
            self.unit = 'q_A^-1'

        # Perform 1D integration
        # qaxis_1d, intensity_1d = pg.integrate_1d(dataarray.data,
        intensity_1d, qaxis_1d = pg.integrate_1d(dataarray.data,
                                                npt=npt,
                                                method=method,
                                                unit=unit,
                                                mask=mask,
                                                correctSolidAngle=correctSolidAngle,
                                                polarization_factor=polarization_factor)
        
        # Create DataArray for integrated 1D data
        integrate1d_da = xr.DataArray(data=intensity_1d,
                                    dims=['qr'],
                                    coords={'qr': ('qr', qaxis_1d, {'units': '1/Å'})},
                                    attrs=dataarray.attrs)
        
        self.integrate1d_da = integrate1d_da
        self.intensity_1d = intensity_1d
        self.qaxis_1d = qaxis_1d
        
        return self.integrate1d_da
        # return intensity_1d, qaxis_1d

    def pyFAI_integrate1D(self, 
                          dataarray, 
                          npt=2250,
                          correctSolidAngle=True,
                          method=("full", "histogram", "cython"),
                          polarization_factor=None, 
                          dark=None, 
                          flat=None,
                          mask = None,
                          radial_range=None, 
                          azimuth_range=None
                        #   unit="2th_deg"
                        #   unit="q_A^-1"
                        #   filename=None,
                        #   mask=None, 
                        #   method="csr", 
                        #   variance=None, 
                        #   error_model=None,
                        #   dummy=None, 
                        #   delta_dummy=None, 
                        #   safe=True,
                        #   normalization_factor=1.0,
                        #   metadata=None
                        ):
            
        """Calculate the azimuthal integration (1d) of a 2D image.

        Multi algorithm implementation (tries to be bullet proof), suitable for SAXS, WAXS, ... and much more
        Takes extra care of normalization and performs proper variance propagation.

        :param ndarray data: 2D array from the Detector/CCD camera
        :param int npt: number of points in the output pattern
        :param str filename: output filename in 2/3 column ascii format
        :param bool correctSolidAngle: correct for solid angle of each pixel if True
        :param ndarray variance: array containing the variance of the data.
        :param str error_model: When the variance is unknown, an error model can be given: "poisson" (variance = I), "azimuthal" (variance = (I-<I>)^2)
        :param radial_range: The lower and upper range of the radial unit. If not provided, range is simply (min, max). Values outside the range are ignored.
        :type radial_range: (float, float), optional
        :param azimuth_range: The lower and upper range of the azimuthal angle in degree. If not provided, range is simply (min, max). Values outside the range are ignored.
        :type azimuth_range: (float, float), optional
        :param ndarray mask: array with  0 for valid pixels, all other are masked (static mask)
        :param float dummy: value for dead/masked pixels (dynamic mask)
        :param float delta_dummy: precision for dummy value
        :param float polarization_factor: polarization factor between -1 (vertical) and +1 (horizontal).
            0 for circular polarization or random,
            None for no correction,
            True for using the former correction
        :param ndarray dark: dark noise image
        :param ndarray flat: flat field image
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
        :param Unit unit: Output units, can be "q_nm^-1" (default), "2th_deg", "r_mm" for now.
        :param bool safe: Perform some extra checks to ensure LUT/CSR is still valid. False is faster.
        :param float normalization_factor: Value of a normalization monitor
        :param metadata: JSON serializable object containing the metadata, usually a dictionary.
        :return: Integrate1dResult namedtuple with (q,I,sigma) +extra informations in it.
        """
                          
        # ai = pyFAI.load(self.poniPath)
        ai = pyFAI.load(str(self.poniPath))

        # Load Mask
        mask = self.load_mask()

        int1d_array = ai.integrate1d_ng(dataarray.data,
                                            npt=npt,
                                            correctSolidAngle=correctSolidAngle,
                                            method=method,
                                            polarization_factor=polarization_factor, 
                                            dark=dark, 
                                            flat=flat,
                                            radial_range=radial_range, 
                                            azimuth_range=azimuth_range,
                                            mask = mask)
                                            # unit=unit)

        # print(method)
        return int1d_array