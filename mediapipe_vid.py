import cv2
import mediapipe as mp
import numpy as np
import struct
from multiprocessing import shared_memory, Process, Pipe

SHOW_INDEX = True

BUFFER_SIZE = 1

IMG_PROPERTIES = {
    "size": 921600,
    "shape": (480, 640, 3),
    "dtype": np.uint8,
    "rate": 30
}

class Recorder:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def readFrame(self):
        success, frame = self.cap.read()
        if success:
            return frame
        return None

    def __del__(self):
        self.cap.release()

class SharedMemoryInterface:
    def __init__(self, shm_names: list):
        buffer_size_bytes = BUFFER_SIZE * IMG_PROPERTIES["size"]
        index_size_bytes = BUFFER_SIZE * 4

        # If shm doesn't exist, set create to True, otherwise, set create to False
        try:
            self.img_shm = shared_memory.SharedMemory(name=shm_names[0], create=True, size=buffer_size_bytes)
            self.index_shm = shared_memory.SharedMemory(name=shm_names[1], create=True, size=index_size_bytes)
        except FileExistsError:
            self.img_shm = shared_memory.SharedMemory(name=shm_names[0], create=False, size=buffer_size_bytes)
            self.index_shm = shared_memory.SharedMemory(name=shm_names[1], create=False, size=index_size_bytes)
    
    def putFrame(self, frame: np.ndarray, index: int):
        # Define buffer indices
        start_idx_frame = (index % BUFFER_SIZE) * IMG_PROPERTIES["size"]
        end_idx_frame = start_idx_frame + IMG_PROPERTIES["size"]
        start_idx_index = (index % BUFFER_SIZE) * 4
        end_idx_index = start_idx_index + 4

        # Write to buffer
        img_buffer = self.img_shm.buf[start_idx_frame:end_idx_frame]
        img_shm_array = np.ndarray(IMG_PROPERTIES["shape"], dtype=IMG_PROPERTIES["dtype"], buffer=img_buffer)
        np.copyto(img_shm_array, frame)

        self.index_shm.buf[start_idx_index:end_idx_index] = struct.pack('i', index)

    def getFrame(self, index: int):
        # Define buffer indices
        start_idx_frame = (index % BUFFER_SIZE) * IMG_PROPERTIES["size"]
        end_idx_frame = start_idx_frame + IMG_PROPERTIES["size"]
        start_idx_index = (index % BUFFER_SIZE) * 4
        end_idx_index = start_idx_index + 4
        
        # Read from buffer
        img_buffer = self.img_shm.buf[start_idx_frame:end_idx_frame]
        img_shm_array = np.ndarray(IMG_PROPERTIES["shape"], dtype=IMG_PROPERTIES["dtype"], buffer=img_buffer)

        index_buffer = self.index_shm.buf[start_idx_index:end_idx_index]
        index_value = struct.unpack('i', index_buffer)[0]

        return img_shm_array, index_value
    
    def __del__(self):
        try:
            self.img_shm.close()
            self.index_shm.close()
            self.img_shm.unlink()
            self.index_shm.unlink()
        except FileNotFoundError:
            pass
    
class MediaPipeEstimator:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.green_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))  # Green color in BGR
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def runOne(self, frame):
        frame.flags.writeable = False
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=None,
                    landmark_drawing_spec=self.green_spec,
                    connection_drawing_spec=None)
        return image    

def recordLoop(mp_conn):
    index = 0
    rec_mp_mem = SharedMemoryInterface(["rec_mp_img", "rec_mp_index"])
    recorder = Recorder()

    # Read frames from camera, put into buffer
    while recorder.cap.isOpened():
        frame = recorder.readFrame()
        rec_mp_mem.putFrame(frame, index)
        index += 1
        mp_conn.send("R")

def mediapipeLoop(rec_conn, disp_conn):
    rec_mp_mem = SharedMemoryInterface(["rec_mp_img", "rec_mp_index"])
    mp_disp_mem = SharedMemoryInterface(["mp_disp_img", "mp_disp_index"])
    est = MediaPipeEstimator()
    index = 0

    # Wait for first camera frame, then run mediapipe
    rec_conn.recv()
    while True:
        frame_from_mem, idx_from_mem = rec_mp_mem.getFrame(index)
        frame_mp = est.runOne(frame_from_mem)
        mp_disp_mem.putFrame(frame_mp, idx_from_mem)
        index += 1
        disp_conn.send("S")
        rec_conn.recv()

def displayLoop(mp_conn):
    mp_disp_mem = SharedMemoryInterface(["mp_disp_img", "mp_disp_index"])
    index = 0

    # Wait for first mediapipe frame, then display
    mp_conn.recv()
    while True:
        frame_from_mem, idx_from_mem = mp_disp_mem.getFrame(index)
        if SHOW_INDEX:
            index_text = f"{idx_from_mem}"
            cv2.putText(frame_from_mem, index_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("MediaPipe Landmarks", frame_from_mem)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        index += 1
        mp_conn.recv()

if __name__ == "__main__":
    # Initialize pipes 
    mp_conn, rec_conn = Pipe()
    disp_conn, mp_conn2 = Pipe()

    # Intialize processes
    p1 = Process(target=recordLoop, args=(mp_conn, ))
    p2 = Process(target=mediapipeLoop, args=(rec_conn, disp_conn, ))
    p3 = Process(target=displayLoop, args=(mp_conn2, ))

    processes = [p1, p2, p3]
    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()