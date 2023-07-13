import cv2


def convert_to_black_and_white(input_path, output_path):
    # Open the video file
    video = cv2.VideoCapture(input_path)

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )  # Use appropriate codec based on the file extension
    output_video = cv2.VideoWriter(
        output_path, fourcc, fps, (width, height), isColor=False
    )

    # Process each frame of the video
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Write the grayscale frame to the output video file
        output_video.write(gray_frame)

        # Display the resulting frame (optional)
        cv2.imshow("Black and White Video", gray_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the resources
    video.release()
    output_video.release()
    cv2.destroyAllWindows()


# Usage example
input_file = "burette.mp4"
output_file = "BGRBurette.mp4"
convert_to_black_and_white(input_file, output_file)
