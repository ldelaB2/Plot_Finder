import logging

class pf_logger:
    def __init__(self, log_path):
        self.clear_log(log_path)
        self.create_loggers(log_path)

    def clear_log(self, log_path):
        with open(log_path, 'w') as f:
            f.write("")

    def create_loggers(self, log_path):
        self.pre_processing = logging.getLogger("pre_processing")
        self.pre_processing.setLevel(logging.DEBUG)

        self.fft_processing = logging.getLogger("image_processing")
        self.fft_processing.setLevel(logging.DEBUG)

        self.wavepad_processing = logging.getLogger("wavepad_processing")
        self.wavepad_processing.setLevel(logging.DEBUG)

        self.find_plots = logging.getLogger("find_plots")
        self.find_plots.setLevel(logging.DEBUG)

        self.optimize_plots = logging.getLogger("optimize_plots")
        self.optimize_plots.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.pre_processing.addHandler(file_handler)
        self.pre_processing.addHandler(stream_handler)

        self.fft_processing.addHandler(file_handler)
        self.fft_processing.addHandler(stream_handler)

        self.wavepad_processing.addHandler(file_handler)
        self.wavepad_processing.addHandler(stream_handler)

        self.find_plots.addHandler(file_handler)
        self.find_plots.addHandler(stream_handler)

        self.optimize_plots.addHandler(file_handler)
        self.optimize_plots.addHandler(stream_handler)

        self.pre_processing.info("Logger initialized")



