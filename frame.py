
import face_detection
from ie_module import InferenceContext
from openvino.inference_engine import IENetwork
import cv2
import os
from datetime import datetime
import numpy as np

from configs import logging, config

class FrameProcessor:
    QUEUE_SIZE = config['queue_size']

    def __init__(self):
        used_devices = set(config['devices'])
        self.config = config
        self.context = InferenceContext()
        context = self.context
        face_detect_config = config['models']['face_detect']

        cpu_ext = face_detect_config['cpu_ext']
        context.load_plugins(used_devices, cpu_ext, "")
        for d in used_devices:
            context.get_plugin(d).set_config({
                "PERF_COUNT": "YES" if config['perf_stats'] else "NO"})

        logging.info("Loading models")
        face_detector_net = self.load_model(face_detect_config['model'])
        

        
        self.face_detector = face_detection.FaceDetector(face_detector_net,
                                          confidence_threshold=face_detect_config['confidence_threshold'],
                                          roi_scale_factor=face_detect_config['roi_scale_factor'])

        

        self.face_detector.deploy(face_detect_config['device'], context)
        

        logging.info("Models are loaded")

    def load_model(self, model_path):
        model_path = os.path.abspath(model_path)
        model_description_path = model_path
        model_weights_path = os.path.splitext(model_path)[0] + ".bin"
        logging.info("Loading the model from '%s'" % (model_description_path))
        assert os.path.isfile(model_description_path), \
            "Model description is not found at '%s'" % (model_description_path)
        assert os.path.isfile(model_weights_path), \
            "Model weights are not found at '%s'" % (model_weights_path)
        model = IENetwork(model_description_path, model_weights_path)
        logging.info("Model is loaded")
        return model

    def identify_image(self, orig_image, source = None):
        frame,rois,landmarks = self.detect_faces(orig_image)
        return self.identify_faces(orig_image,frame,rois,landmarks,source)

    def detect_faces(self,orig_image):
        assert len(orig_image.shape) == 3, \
            "Expected input frame in (H, W, C) format"
        assert orig_image.shape[2] in [3, 4], \
            "Expected BGR or BGRA input"

        frame = orig_image.copy()
        frame = frame.transpose((2, 0, 1))  # HWC to CHW
        frame = np.expand_dims(frame, axis=0)

        self.face_detector.clear()
        self.face_detector.start_async(frame)
        rois = self.face_detector.get_roi_proposals(frame)

        if self.QUEUE_SIZE < len(rois):
            logging.warning("Too many faces for processing."
                            " Will be processed only %s of %s." %
                            (self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]

       
        return rois
