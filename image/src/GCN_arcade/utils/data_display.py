import os
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def display_misclassified_images(misclassified_info, base_path, save_images=False, output_dir=None):
    """
    Display misclassified images and optionally save them to an output directory.

    Args:
        misclassified_info (DataFrame): DataFrame containing information about misclassified images.
        base_path (str): Base path to the image directory.
        save_images (bool, optional): Whether to save the misclassified images. Defaults to False.
        output_dir (str, optional): Directory to save the misclassified images. Defaults to None.
    """
    for index, row in misclassified_info.iterrows():
        image_id_with_idx = row["image_id"]
        image_id = image_id_with_idx[:-2]  # Remove the last two characters from image_id_with_idx
        bbox_idx = image_id_with_idx[-2:]
        image_path = os.path.join(base_path, f"{image_id}_result.png")  # Assuming your images are in PNG format
        image = cv2.imread(image_path)
        
        # Construct the title including information about misclassification
        title = f"Misclassified Image {index + 1}\n"
        title += f"Confidence Score: {row['score']}\n"
        title += f"True Label: {row['label']}\n"
        title += f"Wrongly Classified BBox: {bbox_idx}"

        # Display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        

        # Save the image if save_images is True and output_dir is specified
        if save_images and output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
        
            filename = f"{image_id_with_idx}_Misclassified.png"
            output_path = os.path.join(output_dir, filename)
            # cv2.imwrite(output_path, image)

            # Save the displayed image
            plt.savefig(output_path)
            print(f"Image saved to: {output_path}")

        plt.show()



# 2024/5/16 add
def evaluate_model(model, test_loader, device, task_type='node_classification'):
    y_true = []
    y_pred = []
    y_scores = []

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)

            if task_type == 'graph_classification':
                # Graph classification
                out = model(data.x, data.edge_index, data.batch)
            elif task_type == 'node_classification':
                # Node classification
                out = model(data.x, data.edge_index)
            else:
                raise ValueError("Invalid task_type. Choose either 'node_classification' or 'graph_classification'.")

            prob = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(data.y.cpu().numpy())
            y_scores.extend(prob.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    num_classes = len(np.unique(y_true))

    return y_true, y_pred, y_scores, num_classes



def plot_confusion_matrix(y_true, y_pred, num_classes, exclude_class_1=False):
    if exclude_class_1:
        # ignore class 1
        cm = confusion_matrix(y_true, y_pred, labels=[0, 2])
    else:
        # contain all the class
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=True, square=True)

    label_names = [f"{i}" for i in range(num_classes)]
    if exclude_class_1:
        # if exlcude class 1, reset the label names
        label_names = [label for label in label_names if label != '1']
        tick_marks = [0, 1]
    else:
        tick_marks = range(len(label_names))

    plt.xticks(tick_marks, label_names, rotation=45, fontsize=8)
    plt.yticks(tick_marks, label_names, fontsize=8)

    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.title('Confusion Matrix', fontsize=12)
    plt.show()




def plot_roc_curve(y_true, y_scores, num_classes, binary=False):
    plt.figure()
    if binary:
        # For binary classification, plot a single ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=10)
        plt.ylabel('True Positive Rate', fontsize=10)
        plt.title('ROC Curve', fontsize=12)
        plt.legend(loc='lower right')
        plt.show()
        return roc_auc
    else:
        # For multi-class classification, plot multiple ROC curves
        roc_aucs = []
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            roc_aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve class {i} (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=10)
        plt.ylabel('True Positive Rate', fontsize=10)
        plt.title('ROC Curve', fontsize=12)
        plt.legend(loc='lower right')
        plt.show()
        return roc_aucs