import streamlit as st
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
import math
import csv
from tensorflow.keras.models import load_model

# Load the gesture recognition model
gesture_model = load_model("gesture_recognition_model.h5")

st.markdown("""
    <style>
        .stApp {
    background-color: #040533;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='688' height='688' viewBox='0 0 800 800'%3E%3Cg fill='none' stroke='%230D4344' stroke-width='1'%3E%3Cpath d='M769 229L1037 260.9M927 880L731 737 520 660 309 538 40 599 295 764 126.5 879.5 40 599-197 493 102 382-31 229 126.5 79.5-69-63'/%3E%3Cpath d='M-31 229L237 261 390 382 603 493 308.5 537.5 101.5 381.5M370 905L295 764'/%3E%3Cpath d='M520 660L578 842 731 737 840 599 603 493 520 660 295 764 309 538 390 382 539 269 769 229 577.5 41.5 370 105 295 -36 126.5 79.5 237 261 102 382 40 599 -69 737 127 880'/%3E%3Cpath d='M520-140L578.5 42.5 731-63M603 493L539 269 237 261 370 105M902 382L539 269M390 382L102 382'/%3E%3Cpath d='M-222 42L126.5 79.5 370 105 539 269 577.5 41.5 927 80 769 229 902 382 603 493 731 737M295-36L577.5 41.5M578 842L295 764M40-201L127 80M102 382L-261 269'/%3E%3C/g%3E%3Cg fill='%230F550E'%3E%3Ccircle cx='769' cy='229' r='5'/%3E%3Ccircle cx='539' cy='269' r='5'/%3E%3Ccircle cx='603' cy='493' r='5'/%3E%3Ccircle cx='731' cy='737' r='5'/%3E%3Ccircle cx='520' cy='660' r='5'/%3E%3Ccircle cx='309' cy='538' r='5'/%3E%3Ccircle cx='295' cy='764' r='5'/%3E%3Ccircle cx='40' cy='599' r='5'/%3E%3Ccircle cx='102' cy='382' r='5'/%3E%3Ccircle cx='127' cy='80' r='5'/%3E%3Ccircle cx='370' cy='105' r='5'/%3E%3Ccircle cx='578' cy='42' r='5'/%3E%3Ccircle cx='237' cy='261' r='5'/%3E%3Ccircle cx='390' cy='382' r='5'/%3E%3C/g%3E%3C/svg%3E");
    
    background-attachment: fixed;
    background-size: cover;
        }
    </style>""", unsafe_allow_html=True
)





images_folder = r'C:\\Users\\siva\\Desktop\\images\\'

# Define a mapping from text to image filenames
sign_language_dict = {
    'A': 'A.png',
    'B': 'B.png',
    'C': 'C.png',
    'D': 'D.png',
    'E': 'E.png',
    'F': 'F.png',
    'G': 'G.png',
    'H': 'H.png',
    'I': 'I.png',
    'J': 'J.png',
    'K': 'K.png',
    'L': 'L.png',
    'M': 'M.png',
    'N': 'N.png',
    'O': 'O.png',
    'P': 'P.png',
    'Q': 'Q.png',
    'R': 'R.png',
    'S': 'S.png',
    'T': 'T.png',
    'U': 'U.png',
    'V': 'V.png',
    'W': 'W.png',
    'X': 'X.png',
    'Y': 'Y.png',
    'Z': 'Z.png',
    ' ': 'space.png',
}


def text_to_sign(text):
    images = [Image.open(images_folder + sign_language_dict[char]) for char in text.upper()]
    # Create a GIF from the images
    images[0].save("output.gif", save_all=True, append_images=images[1:], duration=500, loop=0)
    st.image("output.gif", caption="Sign Language GIF", use_column_width=True)

def convert_text_to_sign_language():
    st.title("Text to Sign Language GIF Converter")
    user_input = st.text_input("Enter text:")

    if st.button("Convert to Sign Language"):
        if user_input:
            text_to_sign(user_input)
        else:
            st.warning("Please enter text to convert.")
    stop_button = st.button("Stop Animation")
    if stop_button:
        st.session_state.stop_animation = True

def sign_to_text_module():
    st.markdown("""
        <style>
            .stApp {
        background-color: #040533;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='688' height='688' viewBox='0 0 800 800'%3E%3Cg fill='none' stroke='%230D4344' stroke-width='1'%3E%3Cpath d='M769 229L1037 260.9M927 880L731 737 520 660 309 538 40 599 295 764 126.5 879.5 40 599-197 493 102 382-31 229 126.5 79.5-69-63'/%3E%3Cpath d='M-31 229L237 261 390 382 603 493 308.5 537.5 101.5 381.5M370 905L295 764'/%3E%3Cpath d='M520 660L578 842 731 737 840 599 603 493 520 660 295 764 309 538 390 382 539 269 769 229 577.5 41.5 370 105 295 -36 126.5 79.5 237 261 102 382 40 599 -69 737 127 880'/%3E%3Cpath d='M520-140L578.5 42.5 731-63M603 493L539 269 237 261 370 105M902 382L539 269M390 382L102 382'/%3E%3Cpath d='M-222 42L126.5 79.5 370 105 539 269 577.5 41.5 927 80 769 229 902 382 603 493 731 737M295-36L577.5 41.5M578 842L295 764M40-201L127 80M102 382L-261 269'/%3E%3C/g%3E%3Cg fill='%230F550E'%3E%3Ccircle cx='769' cy='229' r='5'/%3E%3Ccircle cx='539' cy='269' r='5'/%3E%3Ccircle cx='603' cy='493' r='5'/%3E%3Ccircle cx='731' cy='737' r='5'/%3E%3Ccircle cx='520' cy='660' r='5'/%3E%3Ccircle cx='309' cy='538' r='5'/%3E%3Ccircle cx='295' cy='764' r='5'/%3E%3Ccircle cx='40' cy='599' r='5'/%3E%3Ccircle cx='102' cy='382' r='5'/%3E%3Ccircle cx='127' cy='80' r='5'/%3E%3Ccircle cx='370' cy='105' r='5'/%3E%3Ccircle cx='578' cy='42' r='5'/%3E%3Ccircle cx='237' cy='261' r='5'/%3E%3Ccircle cx='390' cy='382' r='5'/%3E%3C/g%3E%3C/svg%3E");
        
        background-attachment: fixed;
        background-size: cover;
            }
        </style>""", unsafe_allow_html=True
    )

    def detect_gesture(frame):
       
        model = load_model("gesture_recognition_model.h5")
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        
        with hands as hands_context:
            results = hands_context.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    thumb_tip = hand_landmarks.landmark[4]
                    index_finger_tip = hand_landmarks.landmark[8]
                    middle_finger_tip = hand_landmarks.landmark[12]
                    ring_finger_tip = hand_landmarks.landmark[16]
                    little_finger_tip = hand_landmarks.landmark[20]

                    if thumb_tip.y < index_finger_tip.y < middle_finger_tip.y < ring_finger_tip.y < little_finger_tip.y:
                        gesture_text = "Okay"
                    elif thumb_tip.y > index_finger_tip.y > middle_finger_tip.y > ring_finger_tip.y > little_finger_tip.y:
                        gesture_text = "Dislike"
                    elif index_finger_tip.y < middle_finger_tip.y and abs(index_finger_tip.x - middle_finger_tip.x) < 0.2:
                        gesture_text = "Victory"
                    elif thumb_tip.x < index_finger_tip.x < middle_finger_tip.x:
                        if (hand_landmarks.landmark[2].x < hand_landmarks.landmark[5].x) and \
                        (hand_landmarks.landmark[3].x < hand_landmarks.landmark[5].x) and \
                        (hand_landmarks.landmark[4].x < hand_landmarks.landmark[5].x):
                            gesture_text = "Stop"
                        else:
                            gesture_text = None
                    elif index_finger_tip.y < middle_finger_tip.y and \
                        thumb_tip.y > index_finger_tip.y and \
                        ring_finger_tip.y > middle_finger_tip.y and \
                        little_finger_tip.y > ring_finger_tip.y:
                        return "Peace"
                    else:
                        wrist = hand_landmarks.landmark[0]
                        index_finger_tip = hand_landmarks.landmark[8]
                        index_finger = (index_finger_tip.x, index_finger_tip.y, index_finger_tip.z)
                        wrist_coords = (wrist.x, wrist.y, wrist.z)
                        vector = (index_finger[0] - wrist_coords[0], index_finger[1] - wrist_coords[1], index_finger[2] - wrist_coords[2])
                        vector_len = (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) ** 0.5
                        vector_unit = (vector[0] / vector_len, vector[1] / vector_len, vector[2] / vector_len)
                        reference_vector = (0, 0, -1)  # the vector pointing towards the camera
                        dot_product = vector_unit[0] * reference_vector[0] + vector_unit[1] * reference_vector[1] + vector_unit[2] * reference_vector[2]
                        angle = math.acos(dot_product) * 180 / math.pi  # angle in degrees
                        if 20 < angle < 80:
                            gesture_text = ""
                        else:
                            gesture_text = None
                        
                    store_landmark_details(thumb_tip, index_finger_tip, middle_finger_tip, ring_finger_tip, little_finger_tip, gesture_text)

                    return gesture_text
        return None

    def store_landmark_details(thumb_tip, index_finger_tip, middle_finger_tip, ring_finger_tip, little_finger_tip, gesture_text):
        landmarks = [thumb_tip, index_finger_tip, middle_finger_tip, ring_finger_tip, little_finger_tip]
        with open("gesture.csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([gesture_text] + [landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks] + [landmark.z for landmark in landmarks])

    st.title('Sign Language Recognition')

    open_camera_button = st.button("Open Camera")
    close_camera_button = st.button("Close Camera")

    cap = None

    if open_camera_button:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                gesture = detect_gesture(frame)
                if gesture:
                    cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                st.image(frame_rgb, channels="BGR")

            if close_camera_button:
                break

    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
def main():
    st.title('🧏🏻‍♀️SIGN LANGUAGE TRANSLATOR🧏🏻')

    page = st.sidebar.selectbox("Select a page", ["Home", "Text to Sign","Sign to Text"])

    if page == "Home":
        
        st.markdown("""
        <style>
        .big-font {
        font-size:50px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<p style='color:yellow;font-weight:bold; font-size: larger; font-size: 20px;' >ABOUT SLT</p>", unsafe_allow_html=True)
        st.write("Welcome to our Sign Language Translator web application! This innovative tool utilizes cutting-edge technology to bridge communication gaps between individuals who use sign language and those who may not be proficient in it. Our web app leverages the power of Convolutional Neural Networks (CNN) and MediaPipe, to accurately detect and translate sign language gestures in real-time.")
        st.write("<span style='color:yellow;font-weight:bold; font-size: larger;'>Features:</span>", unsafe_allow_html=True)
        st.write("1. Real-time Translation: Our web app offers real-time translation of sign language gestures into text, making communication seamless and accessible for everyone.")
        st.write("2. Gesture Recognition: Using a pre-trained CNN model, our app accurately identifies and interprets various sign language gestures, including common signs")
        st.write("3. User-Friendly Interface: With a simple and intuitive interface, users can easily interact with the app by opening and closing the camera to initiate and stop gesture translation.")
        st.write("4. Accessibility: Our goal is to promote inclusivity and accessibility. Whether you're learning sign language, communicating with individuals who are deaf or hard of hearing, or simply curious about sign language, our web app is designed to cater to diverse user needs.")
        st.write("<span style='color:yellow;font-weight:bold; font-size: larger;'>How it Works:</span>", unsafe_allow_html=True)
        st.write("1. Open Camera: Click the Open Camera button to activate the webcam and begin capturing sign language gestures.")
        st.write("2. Gesture Translation: As you perform sign language gestures in front of the camera, our app utilizes advanced machine learning algorithms to recognize and translate them into text in real-time.")
        st.write("3. Close Camera: When you're finished communicating or using the app, simply click the Close Camera button to deactivate the webcam and stop gesture translation.")
        st.write("Experience the future of communication with our Sign Language Translator web application. Start breaking down barriers and fostering meaningful connections today!")
   
    elif page == "Text to Sign":
        convert_text_to_sign_language()
    elif page == "Sign to Text":
         st.write("<div style = 'text-align: justify;'> <span style='color:yellow;font-weight:bold; font-size: larger; text-align: center'>The translator of  sign language gestures into text. To use this module, simply navigate to the Sign Language Recognition section. Once there, you'll find buttons labeled Open Camera and Close Camera. Click Open Camera to activate your webcam and begin capturing sign language gestures. As you perform gestures in front of the camera, the module will detect and translate them into text in real-time. The recognized gestures will be displayed overlaid on the live video feed. When you're finished communicating or using the module, click Close Camera to deactivate the webcam. With clear instructions and intuitive functionality,it makes communication accessible and inclusive for everyone.</span></div>", unsafe_allow_html=True)
         sign_to_text_module()

if __name__ == '__main__':
    main()
