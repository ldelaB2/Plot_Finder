from classes.find_plots import find_plots
from classes.optimize_plots import optimize_plots
from classes.params import pf_params
from classes.logger import pf_logger
import time, os
import numpy as np

class plot_finder_job:
    def __init__(self, param_path):
        self.param_path = param_path
        self.initilize()
        

    def initilize(self):
        logger_path = os.path.join(os.path.dirname(self.param_path), "plot_finder.log")
        default_param_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "default_params.json")
        self.loggers = pf_logger(logger_path)
        self.params = pf_params(default_param_path, self.param_path, self.loggers.pre_processing)

    def run(self):
        if self.params.user_params["find_plots"] == True:
            find_plots_start_time = time.time()

            # Finding the plots
            find_plots(self.params.user_params, self.loggers)

            # End time
            end_time = time.time()
            self.loggers.find_plots.info(f"Total time for finding plots: {np.round(end_time - find_plots_start_time)}")

        if self.params.user_params["optimize_plots"] == True:
            optimizing_start_time = time.time()

            # Optimizing the plots
            optimize_plots(self.params.user_params, self.loggers)

            # End time
            end_time = time.time()
            self.loggers.optimize_plots.info(f"Total time for optimizing plots: {np.round(end_time - optimizing_start_time)}")

        self.print_plot_finder_logo()

        return

    def print_plot_finder_logo(self):
        # Plotting the log
        banner = [
                "              ____   __        __     ______ _             __           ",
                "             / __ \ / /____   / /_   / ____/(_)____   ____/ /___   _____",
                "            / /_/ // // __ \ / __/  / /_   / // __ \ / __  // _ \ / ___/",
                "           / ____// // /_/ // /_   / __/  / // / / // /_/ //  __// /    ",
                "          /_/    /_/ \____/ \__/  /_/    /_//_/ /_/ \__,_/ \___//_/     "
        ]
        
        logo = [
                "%%%%%%%%%%%%%%%%%%%%%%%%%%&&&&&&&&&&&&&&&&&&&&&&@@@@@@@@@@@@@@@@@@@@@@@&&&&&&&&@",
                "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%&&&&&&&&&&&&&&&&&@@@@@@@@@@@@@@@@@@@@@@@@&&&&&&&&&",
                "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%&&&&&&&&&&&&&&&&@@@@@@@@@@@@@@@@@@@@@@@&&&&&&&&&&",
                "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%&&&&&&,,,,,,,%&@@@@@@@@@@@@@@@@@@@@@@@&&&&&&&&&&&",
                "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%&&&,,,,@@@@@,,,,&@@@@@@@@@@@@@@@@@@@@@&&&&&&&&&&&",
                "%%%%%%%%%%%%%%%%%%%%%%%%%%%%*,,,,,,@@@@@,@@@@&,,,,,,/&@@@@@@@@@@@@@@&&&&&&&&&&&&",
                "%%%%%%%%%%%%%%%%%%%%%%%%%,,,@@@@@@@@@@,@@@,@@@@@@@@@@,,,&&@@@@@@@@@&&&&&&&&&&&&&",
                "%%%%%%%%%%%%%%%%%%%%%%%,,%@@,@@@@,,&@@@,@,@@@@,,@@@&,@@(,,&&@@@@@@@&&&&&&&&&&&&&",
                "%%%%%%%%%%%%%%%%%%%%%/,,@@,@@@@@,,/@&@@,@,@@(@%,/@@@@@,@@,,#&@@@@@&&&&&&&&&&&&&&",
                "%%%%%%%%%%%%%&&&&&&&*,,@@@@@%@@@@,,,,/@@@@@*,,,,@@@@&@@&@@,,(&&@@&&&&&&&&&&&&&&&",
                "%%%%&&&&&&&&&&&&&&&&,,@@&@@@@@@@*,,,,,,,(,,,,,,,*@@@@@@@#@@,,&&&@&&&&&&&&&&&&&&&",
                "&&&&&&&&&&&&&&&&&&&,,@@,@@@@@@,,,,,,,,,,,,,,,,,,,,,@@@@@@,@@,,%&&&&&&&&&&&&&&&&&",
                "&&&&&&&&&&&&&&&&&&@,,@@@@@@@(,,,,,,,,,,,@,,,,,,,,,,,&@@@@@@@,,%%&&&&&&&&&&&&&&&&",
                "&&&&&&&&&&&&&&&&&&#,,@/%@@@@,,@@@@@,,,,(@,,,,,,@@@@,,@@@@%#@,,%%%&&&&&&&&&&&&&&&",
                "&&&&&&&&&&&&&&&&&&*,(@,@@@@,,,,@@@,,,,,@@@,,,,,,@*,,,,@@@@*@*,/#%&&&&&&&&&&&&&&&",
                "&&&&&&&&&&&&&&&&&&/,/@,@@@@,,,,,@@#,,,,@@@,,,,,@@,,,,,@@@@/@,,(#%&&&&&&&&&&&&&&&",
                "&&&&&&&&&&&&&&&&&&%,,@&@@@@,,,,,#@@,,,@@@@@,,,#@,,,,,,@@@@@@,,%%%&&&&&&&&&&&&&&&",
                "&&&&&&&&&&&&&&&&&%%,,@@,@@@,,,,,,@@@,,@@(@@,,,@@,,,,,*@@@,@@,,#%%&&&&&&&&&&&&&&&",
                "&&&&&&&&&&&&&&&&&%%,,@@,@@@@,,,,,,@@,*@@,@@@,@@,,,,,,@@@@,@&,,#%&&&&&&&&&&&&&&&&",
                "&&&&&&&%%%%%%%%%%%%%,,@@,@@@(,,,,,@@@@@,,#@@,@@,,,,,%@@@,@@,,##%&&&&&&&&&&&&&&&&",
                "&&&&&&&&&%%%%%%%%%%%,,/@&@@@@,,,,,,@@@@,,,@@@@,,,,,,@@@@@@,,*#%&&&&&&&&&&&&@&&&&",
                "&&&&&&&&&&%%%%%%%%%%%,,@@*@@@@/,,,,@@@*,,,#@@@,,,,(@@@@(@@,,#%%&&&&&&&&&@@@@@&&&",
                "&&&&&&&&&&&%%%%%%%%%%%,,@@*@@@@,,,,,@@,,,,,@@,,,,,@@@@#@@,,#%%&&&&&&&&@@@@@@@@&&",
                "&&&&&&%%%&&&&%%%%%%%%%%,,@@@&@@@@,,,*@,,,,,%(,,*@@@@*@@%,,#%%&&&&&&@@@@@@@@@@@@@",
                "&&&%%%%%%%%%&&%%%%%%%%%%,,,@@,@@@@@,,,,,,,,,,,@@@@@,@@,,/#%%&&&&&@@@@@@@@@@@@@@@",
                "%%%%%%%%%%%%%&&&%%%%%%%%%#,,%@@,@@@@@@,,,,,@@@@@@,@@(,,##%&&&&@@@@@@@@@@@@@@@@@@",
                "%%%%%%%%&&&&&&&&&%%%%%%%%%%%,,,@@(&@,@@@@@@@,@(%@@,,,##%%&&&@@@@@@@@@@@@@@@@@@@@",
                "%%%%&&&&&&&&&&&&&&%%%%%%%%%%%#,,,#@@@*,@@@,*@@@(,,,##%%&&&&&&&&&&@@@@@@@@@@@@@@@",
                "&&&&&&&&&&&&&&&&&&&&%%%%%%%%%%%##,,,,,@#@*&,,,,*##%%&&&&&&&&&&%%%%%%%@@@@@@@@@@@",
                "&&&&&&&&&&&&&&&&&&&&&%%%%%%%%%%%%%%%%%#,,,####%%%&&&&&&&&&&&&%%%%%%%%%%%%%@@@@@@"
        ]
        
        
        for element in logo:
            print(element)
        print("\n")
        for element in banner:
            print(element)
        print("\n")
        
        print(f"""Finished Processing Image: {self.params.user_params['image_name']}
        Thanks for using PLot Finder! Keep on Keeping on - Squid Billy Willy""")
                                                           