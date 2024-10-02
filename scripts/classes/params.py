import json, os, multiprocessing, rasterio
from functions.pre_processing import compute_GSD, compute_gray_weights, compute_gray, compute_theta
from functions.image_processing import rotate_img
import cv2 as cv

class pf_params:
    def __init__(self, default_param_path, user_param_path, logger):
        self.logger = logger
        self.user_params = self.read_params(user_param_path)
        self.default_params = self.read_params(default_param_path)
        self.create_params()
        
    
    def create_params(self):
        self.check_user_params()
        self.check_output_directory()
        self.check_num_cores()
        self.check_metadata()
        self.check_GSD()
        self.check_image()
        self.check_theta()
        self.check_gray_method()
        

    def read_params(self, param_path):
        try:
            with open(param_path) as f:
                params = json.load(f)
        except FileNotFoundError:
            self.logger.critical(f"User params file not found at: {param_path} Exiting...")
            exit(1)
        except json.JSONDecodeError:
            self.logger.critical(f"User params file is not a valid json file at: {param_path} Exiting...")
            exit(1)
        except Exception as e:
            self.logger.critical(f"Unknown error reading user params file at: {param_path}. Error: {e} Exiting...")
            exit(1)

        return params

    def check_user_params(self):
        if self.user_params["ortho_path"] is None:
            self.logger.critical("Ortho path not found in user params. Exiting...")
            exit(1)
        if self.user_params["output_directory"] is None:
            self.logger.critical("Output directory not found in user params. Exiting...")
            exit(1)
        if self.user_params["number_rows"] is None:
            self.logger.critical("Number of rows not found in user params. Exiting...")
            exit(1)
        if self.user_params["number_ranges"] is None:
            self.logger.critical("Number of ranges not found in user params. Exiting...")
            exit(1)
        if self.user_params["row_spacing_in"] is None:
            self.logger.critical("Row spacing not found in user params. Exiting...")
            exit(1)
        
        self.logger.info("Required user params found")

        # Get the image name from the image path
        self.user_params["image_name"] = os.path.splitext(os.path.basename(self.user_params["ortho_path"]))[0]

        # Pull all defalut params not in user params
        for param, default_value in self.default_params.items():
            if param not in self.user_params:
                self.user_params[param] = default_value

    def check_output_directory(self):
        def create_dir(path, logger):
            try:
                os.makedirs(path, exist_ok = True)
                logger.info(f"Created directory: {path}")
            except Exception as e:
                logger.critical(f"Error creating directory: {path}. Error: {e}")
                exit(1)

        # Create the main output directory
        create_dir(self.user_params["output_directory"], self.logger)

        # Create the image output directory
        img_directory = os.path.join(self.user_params["output_directory"], self.user_params["image_name"] + "_pf_output")
        create_dir(img_directory, self.logger)

        # Create the output directory dictionary
        self.output_directories = {}

        # Create the plots output directory
        if self.user_params["save_plots"] == True:
            plots_directory = os.path.join(img_directory, "plots")
            self.output_directories["plots"] = plots_directory
            create_dir(plots_directory, self.logger)

        # Create the quality output directory
        if self.user_params["QC_depth"] != "none":
            quality_directory = os.path.join(img_directory, "QC")
            self.output_directories["quality"] = quality_directory
            create_dir(quality_directory, self.logger)
        
        shape_directory = os.path.join(img_directory, "shapefiles")
        self.output_directories["shapefiles"] = shape_directory
        create_dir(shape_directory, self.logger)

        if self.user_params["optimize_plots"] == True and self.user_params["save_optimization_model"] == True:
            optimize_models_directory = os.path.join(img_directory, "optimization_models")
            self.output_directories["optimization_models"] = optimize_models_directory
            create_dir(optimize_models_directory, self.logger)

        self.user_params["pf_output_directorys"] = self.output_directories
        
    def check_num_cores(self):
        if self.user_params["num_cores"] == "AUTO":
            self.user_params["num_cores"] = multiprocessing.cpu_count()

            if self.user_params["num_cores"] == "AUTO":
                self.user_params["num_cores"] = os.cpu_count()

                if self.user_params["num_cores"] == "AUTO":
                    self.user_params["num_cores"] = 1
                    self.logger.warning("Number of cores set to: 1")
            
        self.logger.info(f"Number of cores set to: {self.user_params['num_cores']}")

        # TODO add some constrait on numpy, skimage, and rasterio to use the number of cores

    def check_metadata(self):
        if self.user_params["meta_data"] is None:
            with rasterio.open(self.user_params["ortho_path"]) as src:
                self.user_params["meta_data"] = src.meta

            self.logger.info("Extracting metadata from ortho image")
        else:
            self.logger.info("Using existing metadata from params")

    def check_GSD(self):
        if self.user_params["GSD"] == "AUTO":
            self.logger.info("Calculating GSD")
            calculated_gsd = compute_GSD(self.user_params["meta_data"], self.logger)
            self.user_params["GSD"] = calculated_gsd
            self.logger.info(f"Calculated GSD (cm): {calculated_gsd}")
        else:
            self.logger.info(f"Using GSD from params (cm): {self.user_params['GSD']}")

    def check_image(self):
        try :
            img = cv.imread(self.user_params["ortho_path"])
            if img is not None:
                self.logger.info(f"Reading image at: {self.user_params['ortho_path']}")
                self.user_params["img_ortho"] = img
                self.user_params["img_ortho_shape"] = img.shape
            else:
                self.logger.critical(f"Error reading image at: {self.user_params['ortho_path']}. Image is None. Exiting...")
                exit(1)
        except Exception as e:
            self.logger.critical(f"Error reading image at: {self.user_params['ortho_path']}. Error: {e}")
            exit(1)

    def check_gray_method(self):
        if self.user_params["custom_grayscale"] == True:
            self.logger.info(f"Using custom grayscale method {self.user_params['gray_method']}")
        else:
            self.user_params["custom_grayscale"] = False

            gray_method = self.user_params["gray_scale_method"].upper()
            valid_method = ['AUTO', 'BI', 'SCI', 'GLI', 'HI', 'NGRDI', 'SI', 'VARI', 'BGI','GRAY','LAB','HSV']

            if gray_method not in valid_method:
                self.logger.warning(f"Invalid grayscale method: {gray_method}. Using LAB")
                gray_method = 'LAB'
            else:
                self.logger.info(f"Using grayscale method: {gray_method}")

            if gray_method == 'AUTO':
                self.logger.info(f"Using grayscale features of {self.user_params['auto_gray_features']}")
                self.logger.info(f"Using polynomial features of degree {self.user_params['auto_gray_poly_features_degree']}")

                if self.user_params["recompute_auto_gray_weights"] == True or self.user_params["auto_gray_weights"] is None:
                    self.logger.info("Computing auto gray weights")
                    gray_weights = compute_gray_weights(self.user_params, self.logger)
                    self.logger.info(f"Computed gray weights: {gray_weights}")
                    self.user_params["auto_gray_weights"] = gray_weights
                    self.user_params["gray_scale_invert"] = False

        if self.user_params["gray_scale_invert"] == True:
            self.logger.info("Gray scale will be inverted")
        else:
            self.logger.info("Gray scale will not be inverted")
            self.user_params["gray_scale_invert"] = False
        

        # Compute the gray image
        self.logger.info("Computing gray image")

        # Compute the gray image
        custom_flag = self.user_params["custom_grayscale"]
        method = self.user_params["gray_scale_method"]
        invert = self.user_params["gray_scale_invert"]
        image = self.user_params["img_ortho"]
        gray_img = compute_gray(custom_flag, method, image, invert, self.logger)
        self.user_params["gray_img"] = gray_img

        self.logger.info("Finished computing gray image")

    def check_theta(self):
        user_choice = self.user_params["rotation_angle"]
        if user_choice is None:
            self.logger.info("Rotation angle not found in user params. Setting to AUTO")
            user_choice = "AUTO"
        
        if user_choice == "AUTO":
            self.logger.info("Computing rotation angle")
            theta = compute_theta(self.user_params, self.logger)
            self.logger.info(f"Computed rotation angle: {theta} degrees")
            self.user_params["rotation_angle"] = theta

        else:
            self.logger.info(f"Using user defined rotation angle: {user_choice}")
            theta = user_choice

        self.user_params["inverse_rotation_matrix"], self.user_params["rotation_matrix"], self.user_params["img_ortho"] = rotate_img(self.user_params["img_ortho"], theta) 
        self.logger.info("Finished rotating image")
        


        

 