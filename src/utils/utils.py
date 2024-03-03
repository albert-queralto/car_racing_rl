import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from base64 import b64encode
from IPython.display import HTML


def preprocess_observation(observation: np.ndarray) -> np.ndarray:
    """Preprocesses the observation from the environment."""
    # Converts the image to grayscale
    observation =  cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

    # Resizes the image to 84x84 pixels
    # observation = observation[:84, 6:90]
    observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
    
    return np.array(observation).astype(np.float32) / 255.0 # Normalizes the image

def stack_frames(
        stacked_frames: np.array,
        frame: np.array,
        is_new: bool,
        stack_frames: int = 4
    ) -> np.array:
    if is_new:
        stacked_frames = np.stack(arrays=[frame for _ in range(stack_frames)], axis=0)
    else:
        stacked_frames = np.roll(stacked_frames, shift=-1, axis=0)
        stacked_frames[-1] = frame
    return stacked_frames

class VideoUtils:
    """
    Utility class for video rendering in Jupyter Notebook.
    """
    def __init__(self, video_path):
        """
        Constructor of the VideoUtils class.
        
        Parameters
        ----------
        video_path: str
            Path to the video file.
        """
        self.video_path = video_path

    def _reproduce_html_(self):
        """
        Reproduces a video in an HTML embedded object in Jupyter Notebook.
        It reads the content of the video file and encodes it in base64 format
        so that it can be later decoded in the return html string.
        
        Returns
        -------
        str
            HTML representation of the video.
        """
        video = open(self.video_path, "rb").read()
        video_encoded = b64encode(video).decode("ascii")
        return f"""
        <video width="600" height="400" controls autoplay>
            <source src="data:video/mp4;base64,{video_encoded}" type="video/mp4">
        </video>
        """
        
    def _reproduce_opencv_(self):
        """
        Reproduces a video in Jupyter Notebook using the OpenCV library.
        It creates a capture object that is used to read the frames from the
        video. For every read frame, it shows the frame in a window every 25 ms.
        If the 'q' key from the keyboard is press, the video is stopped.
        Finally, on the video is finished, releases the memory and closes the
        window.
        
        Returns
        -------
        None
        """
        cap = cv2.VideoCapture(self.video_path)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                cv2.imshow('frame',frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
            
        cap.release()
        time.sleep(1)
        cv2.destroyAllWindows()


    def show_html(self):
        """
        Displays the video in Jupyter Notebook by calling the _reproduce_html_
        method. It wraps the video in an embedded HTML object.
        """
        return HTML(self._reproduce_html_())


    def show_opencv(self):
        """
        Displays the video in Jupyter Notebook by calling the _reproduce_opencv_
        method.
        """
        return self._reproduce_opencv_()
    

class PlotUtils:
    """
    Utility class that provides methods to plot the results from PyTorch models.
    It takes a dataframe that is used to build the plots.
    """
    def __init__(self, df):
        """
        Constructor of the PlotUtils class.
        
        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with the results of the models.
        """
        self.df = df


    def _create_labels(self):
        """
        Creates the labels for the plots using the parameters of dataframe rows.
        This is done by filtering the original dataframe by the model column,
        thus extracting the unique models. Creates a label for each model by
        concatenating the values of the different hyperparameters. It also
        considers special cases such as the parameters used for the
        PrioritizedExperienceReplay buffer or the N-step Bellman equation.
        """
        labels_df = self.df.drop_duplicates(subset=['model'])
        labels = []
        for _, row in labels_df.iterrows():
            label = (
                        f"model{row['model']}_" +
                        f"nb{row['nblock']}_" +
                        f"bs{row['batch_size']}_" + 
                        f"lr{row['learning_rate']}_" +
                        f"g{row['gamma']}_" +
                        f"uf{row['update_frequency']}_" +
                        f"sf{row['sync_frequency']}_" +
                        f"hl{row['hidden_layers']}_" +
                        f"ed{row['epsilon_decay']}"
                    )
            # Add alpha, beta and beta_increment if they are present in df.columns
            # and the value is not NaN
            if 'alpha' in self.df.columns and not pd.isnull(row['alpha']):
                label += f"_a{row['alpha']}"
            if 'beta' in self.df.columns and not pd.isnull(row['beta']):
                label += f"_b{row['beta']}"
            if 'beta_increment' in self.df.columns and not pd.isnull(row['beta_increment']):
                label += f"_bi{row['beta_increment']}"
            # Add n_step if it is present in df.columns and the value is not NaN
            if 'n_step' in self.df.columns and not pd.isnull(row['n_step']):
                label += f"_ns{row['n_step']}"            
            labels.append(label)
        return labels_df, labels


    def compare_hyperparams(self):
        """
        Plots the results of the models for different hyperparameters. It first
        creates the labels using the _create_labels method. Then, creates a
        figure with 3 plots. The first plot shows the evolution of episode 
        average rewards during training and the threshold value. The second 
        shows the evolution of the loss. Finally, the third plot shows the decay 
        of epsilon.

        Returns:
        -------
        Plots with the rewards, loss and epsilon for different models.
        """
        
        # Set labels
        axes_labels = ['Average Reward', 'Loss', 'Epsilon']
        labels_df, labels = self._create_labels()
        
        # Create plots
        _, axes = plt.subplots(1,3, figsize=(40,7))

        for i in range(3):
            if i == 0:
                # Plot the average_rewards and threshold
                for _, row in labels_df.iterrows():
                    axes[i].plot(self.df[self.df['model'] == 
                                                    row['model']]['game_count'], 
                                    self.df[self.df['model'] == 
                                                row['model']]['average_rewards'])
                axes[i].axhline(y=200, label='Reward threshold', color='black', 
                                                    linestyle='--', alpha=1.0)

            if i == 1:
                # Plot the loss
                for _, row in labels_df.iterrows():
                    axes[i].plot(self.df[self.df['model'] == 
                                                    row['model']]['game_count'], 
                                self.df[self.df['model'] == 
                                                row['model']]['loss_evolution'])
            
            if i == 2:
                # Plot the epsilon
                for _, row in labels_df.iterrows():
                    axes[i].plot(self.df[self.df['model'] == 
                                                    row['model']]['game_count'], 
                                self.df[self.df['model'] == 
                                                        row['model']]['epsilon'])

            axes[i].set_xlabel('Episodes', fontsize=22)
            axes[i].set_ylabel(axes_labels[i], fontsize=22)
            axes[i].tick_params(axis='both', which='major', labelsize=20)  
            axes[i].grid()

        lines = plt.gca().get_lines()
        colors = []
        for line in lines:
            colors.append(line.get_color())

        plt.legend(labels, loc='upper center', ncol=2, fontsize=18, 
                                                bbox_to_anchor=(-0.8, -0.15))
        plt.show()