from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.awb_mode = 'tungsten'
camera.start_preview(fullscreen=False, window=(0, 0, 1280, 720))
sleep(15)
camera.capture('image-HD.jpg', use_video_port=True)
camera.stop_preview()
