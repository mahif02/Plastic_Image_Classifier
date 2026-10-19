import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Page settings
st.set_page_config(
    page_title="Plastic Image Classifier",
    page_icon="♻️"
)

# Load model
@st.cache_resource
def load_model():
    return YOLO(
        "runs/detect/plastic_project/plastic_detector-6/weights/best.pt"
    )

model = load_model()

# Title
st.title("♻️ Plastic Image Classifier")
st.write("Upload an image to detect plastic waste.")

st.divider()

# Upload image
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    if st.button("🔍 Predict", use_container_width=True):

        # Run prediction
        results = model.predict(
            source=image,
            conf=0.25
        )

        # Show image with bounding boxes
        result_image = results[0].plot()

        st.subheader("Detection Result")
        st.image(
            result_image,
            use_container_width=True
        )

        # Store predictions
        predictions = []

        st.subheader("📊 Predictions")

        for result in results:

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    class_name = model.names[class_id]

                    predictions.append(class_name)

                    st.success(
                        f"**{class_name}** — "
                        f"{confidence * 100:.2f}% confidence"
                    )

        # Summary
        if predictions:

            st.subheader("📋 Prediction Summary")

            st.write(
                f"**Objects detected:** {len(predictions)}"
            )

            st.write(
                f"**Bottle:** {predictions.count('Bottle')}"
            )

            st.write(
                f"**Bag:** {predictions.count('Bag')}"
            )

            st.write(
                f"**Cup:** {predictions.count('Cup')}"
            )

        else:

            st.warning("No plastic object detected.")