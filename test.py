
from imageai.Detection import ObjectDetection, VideoObjectDetection
import os
import time

def image_object_detection(model_name, pretrained_model, custom):
    start = time.time()

    execution_path = os.getcwd()
    detector = ObjectDetection()

    if model_name == 'RetinaNet':
        detector.setModelTypeAsRetinaNet()
    elif model_name == 'YOLOv3':
        detector.setModelTypeAsYOLOv3()
    elif model_name == 'TinyYOLOv3':
        detector.setModelTypeAsTinyYOLOv3()

    detector.setModelPath(os.path.join(execution_path, 'pretrained_h5files', pretrained_model))
    detector.loadModel()

    if custom:
        custom_obj = detector.CustomObjects(person=True)
        detector.detectCustomObjectsFromImage(
            input_image=os.path.join(execution_path, "dataset/test_image/image.jpg"),
            output_image_path=os.path.join(execution_path, "result/result_image/result_custom_image.jpg"),
            custom_objects=custom_obj, minimum_percentage_probability=50)
    else:
        detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "dataset/test_image/image.jpg"),
                                        output_image_path=os.path.join(execution_path, "result/result_image/result_image.jpg"),
                                        minimum_percentage_probability=50)

    end = time.time()
    print('Total time: %f' % (end - start))



def video_object_detection(model_name, pretrained_model, custom):
    start = time.time()

    execution_path = os.getcwd()
    detector = VideoObjectDetection()

    if model_name == 'RetinaNet':
        detector.setModelTypeAsRetinaNet()
    elif model_name == 'YOLOv3':
        detector.setModelTypeAsYOLOv3()
    elif model_name == 'TinyYOLOv3':
        detector.setModelTypeAsTinyYOLOv3()

    detector.setModelPath(os.path.join(execution_path, 'pretrained_h5files', pretrained_model))
    detector.loadModel()

    if custom:
        custom_obj = detector.CustomObjects(person=True)
        detector.detectCustomObjectsFromVideo(
            input_file_path=os.path.join(execution_path, "dataset/test_video/traffic.mp4"),
            output_file_path=os.path.join(execution_path, "result/result_video/result_custom_traffic"),
            custom_objects=custom_obj, frames_per_second=20, log_progress=True)
    else:
        detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "dataset/test_video/traffic.mp4"),
                                                 output_file_path=os.path.join(execution_path, 'result/result_video/result_traffic'),
                                                 frames_per_second=20, log_progress=True)

    end = time.time()
    print('Total time: %f' %(end-start))


if __name__ == '__main__':
    detection_model_name = 'RetinaNet' #List of detection model : 'RetinaNet', 'YOLOv3', 'TinyYolov3'
    pretrained = 'resnet50_coco_best_v2.1.0.h5'

    # image_object_detection(detection_model_name, pretrained, False)
    # image_object_detection(detection_model_name, pretrained, True)

    # video_object_detection(detection_model_name, pretrained, False)
    # video_object_detection(detection_model_name, pretrained, True)