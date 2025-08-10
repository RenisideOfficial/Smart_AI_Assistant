# Smart AI Assistant for the Visually Impaired

### Project Overview

The Smart AI Assistant is a comprehensive, real-time support tool designed to assist visually impaired individuals. By integrating computer vision, voice recognition, and AI, this assistant provides a range of features to enhance daily life, from navigating the environment to getting weather updates.

---

### Features

- **Voice-Activated Control:** Interact with the assistant using natural speech commands.
- **Real-Time Object Detection:** Uses computer vision to identify and describe objects in your surroundings.
- **Text-to-Speech (TTS):** All text-based information is converted into clear, audible speech.
- **Wayfinding Assistance:** Get turn-by-turn navigation using GPS and mapping services.
- **Public Transport Guidance:** Find nearby bus, train, and subway routes with ease.
- **Live Weather Updates:** Receive up-time weather forecasts for your current location.
- **Intuitive GUI:** A simple and accessible graphical interface built with Tkinter for setup and interaction.

---

### Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV
- **Speech Recognition:** Google Speech API
- **Text-to-Speech:** `pyttsx3`
- **Automation:** `PyAutoGUI`
- **Maps & Navigation:** Google Maps API
- **GUI:** Tkinter
- **Machine Learning:** TensorFlow/PyTorch (for object detection)

---

### Installation

#### Prerequisites

Make sure you have **Python 3.8** or newer installed on your system.

#### Setup

1.  Clone the repository:
    ```
    git clone https://github.com/your-username/Smart-AI-Assistant.git
    cd Smart-AI-Assistant
    ```
2.  Install the necessary dependencies:
    ```
    pip install -r requirements.txt
    ```

---

### Usage

To start the assistant, run the main script from your terminal:

```
python assistant.py
```

#### Example Voice Commands

- "Describe my surroundings"
- "Where is the nearest bus stop?"
- "What's the weather like?"

---

### Project Structure

```
Smart-AI-Assistant/
├── assistant.py            # Core AI assistant logic
├── assistant_gui.py        # Graphical user interface
├── object_detection.py     # OpenCV object detection module
├── weather.py              # Fetches and processes weather data
├── requirements.txt        # Project dependencies
└── README.md
```

---

### Future Enhancements

- **Advanced AI:** Integrate a conversational AI chatbot for more natural interactions.
- **Deep Learning Models:** Upgrade object detection with more accurate and efficient deep learning models.
- **Real-Time Tracking:** Improve GPS navigation with continuous, real-time location tracking.
- **Smart Home Integration:** Add support for controlling smart home devices.

---

### Contribution

We welcome contributions\! Feel free to fork the repository, open issues, or submit pull requests.

---

### Contact

For any questions or feedback, please open an issue on GitHub or email: chinweikeprince95@gmail.com
