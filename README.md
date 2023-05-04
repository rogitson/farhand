# Farhand
This project aims to improve mediapipe performance at a distance by first using blazepose to find the body, cropping the image around the hand, and then using mediapipe-hands on just the cropped image.

The reason this project exists is that mediapipe typically struggles to detect hands past the 2 meter mark. While this may not be an issue depending on your usecase, it proved to be a big issue to me in a different project I was working on. I eventually abandoned this idea in the other project as it required real time processing and the added overhead of blazepose was too much, and I decided to upload my efforts regardless as they may help someone else.

I've personally tested it at about 4-5 meters. It is sometimes dependant on the lighting of the area, and is not always ideal, but for the most part the hand was at least detected whereas it would not have been had it been just the uncropped image. It may not be very accurate when used for hand gestures without further modification. 

Do note that this project will understandably perform with lower fps than just running mediapipe-hands alone because of the blazepose overhead. 

Contributions are welcome.

## Usage
The current provided demo code (main.py) uses the camera at index 0 and will display the camera feed with blazepose annotations, as well as a smaller square image in the bottom left corner that will either be the cropped hand, or if no cropping is happening, a black square.

## Credits
A lot of the code for the hand cropping was taken and modified from https://github.com/geaxgx/depthai_hand_tracker.