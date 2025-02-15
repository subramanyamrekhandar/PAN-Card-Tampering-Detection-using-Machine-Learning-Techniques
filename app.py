import cv2
import numpy as np
import streamlit as st
from skimage.metrics import structural_similarity as ssim
from streamlit_option_menu import option_menu


def process_image(image, reference):
    try:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to a fixed size
        size = (600, 400)
        resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

        # Compute the structural similarity index
        ssim_index = ssim(resized, reference)

        # Apply thresholding to detect tampering
        threshold = 0.8
        tampering_detected = ssim_index < threshold

        # Find contours in the image
        contours, _ = cv2.findContours(resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return tampering_detected, ssim_index, contours

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None, None


def draw_contours(image, contours):
    # Draw the contours on the image
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 100:
            cv2.drawContours(image, contours, i, (0, 0, 255), 2)


# Define the Streamlit app
def main():
    st.set_page_config(page_title='PAN Card Tampering Detection')

    # Sidebar navigation
    # st.sidebar.title("Navigation")
    # page = st.sidebar.radio("Go to", ["Home", "Detection"])
    with st.sidebar:
       page = option_menu(
        menu_title="Main Menu",
        options=["Home", "Detection"],
        icons=["house", "robot"],
        menu_icon="cast",
        default_index=0,
    )


    if page == "Home":
        st.title("Welcome to PAN Card Tampering Detection")
        st.write("Use this tool to detect tampering in PAN card images. Navigate to the **Detection** page to start.")
        st.image("testdata/banner.jpg", caption="Example PAN Card", use_container_width=True)

    elif page == "Detection":
        st.title('PAN Card Tampering Detection')
        st.write('Upload an image of a PAN card to detect tampering and validate the ID.')

        # Load the reference image
        reference_image = cv2.imread('testdata/reference.jpg', cv2.IMREAD_GRAYSCALE)
        if reference_image is None:
            st.error("Reference image not found! Please ensure  is in the same directory.")
            return

        reference_image_resized = cv2.resize(reference_image, (600, 400), interpolation=cv2.INTER_AREA)

        # Add a file uploader to get the image from the user
        uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            try:
                # Read the image
                image_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

                if image is None:
                    st.error("Invalid image file. Please upload a valid image.")
                    return

                # Process the image
                tampering_detected, ssim_index, contours = process_image(image, reference_image_resized)

                if tampering_detected is None:
                    return

                # Draw the contours on the image
                draw_contours(image, contours)

                # Display the image and results
                st.image(image, channels='BGR', use_column_width=True)
                if tampering_detected:
                    st.error('Tampering detected!')
                else:
                    st.success('No tampering detected.')
                st.write(f'Structural similarity index: {ssim_index:.2f}')

            except Exception as e:
                st.error(f"Error processing image: {e}")


if __name__ == '__main__':
    main()
