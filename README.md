# Emotion_Detection

**Uncover the emotions behind expressions. Analyze faces in real-time using this deep learning project.**

**What it does:**

- Classifies emotions into **seven categories:** angry, disgusted, fearful, happy, neutral, sad, and surprised.
- Employs a convolutional neural network (CNN) trained on the **FER-2013** dataset.
- Detects emotions from your webcam feed or images.

**Key features:**

- **Lightweight and efficient:** Runs on standard computers with basic GPUs.
- **Pre-trained model included:** Get started easily without training.
- **Customizable:** Train on your own datasets for unique use cases.
- **Open-source and easy to use:** Contribute to the code and make it your own.

**Get started:**

**Clone the repository:**

   ```bash
   git clone [https://github.com/atulapra/Emotion-detection.git](https://github.com/atulapra/Emotion-detection.git)
   cd Emotion-detection
   ```

* Download the FER-2013 dataset inside the `src` folder.

* If you want to train this model, use:  

```bash
cd src
python emotions.py --mode train
```

```bash
cd src
python emotions.py --mode display
```

* The folder structure is of the form:  
  src:
  * data (folder)
  * `emotions.py` (file)
  * `haarcascade_frontalface_default.xml` (file)
  * `model.h5` (file)


