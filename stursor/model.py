import torch
from nnsight import LanguageModel
import matplotlib.pyplot as plt
import openai

class Model:
    def __init__(self, model_name):
        self.model = LanguageModel(
            model_name,
            device_map="auto",
            device=torch.bfloat16
        )

    def generate_text(self, prompt):
        return self.model.generate(prompt)

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.show()

    def plot_accuracy(self):
        plt.plot(self.accuracy_history)
        plt.show()