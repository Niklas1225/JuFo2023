# <b>Alzheimer-Erkennung durch Künstliche Intelligenz</b> - JuFo2023

<h3>Kurzfassung</h3>
<hr>

__German__ <br>
<span><em>
Alzheimer ist eine zunehmend auftretende neurodegenerative Erkrankung. In ihrem Verlauf nimmt die Fähigkeit des Gehirns Informationen zu speichern ab, die Sprachfähigkeit leidet. Durch eine frühe Erkennung kann das Fortschreiten der Erkrankung entscheidend verlangsamt werden, sodass für Betroffene und Angehörige eine bessere Situation geschaffen wird.<br>
In meinem Projekt möchte ich mittels KI-basierter Ansätze die Diagnostik von Alzheimer in allen Stadien der Erkrankung verbessern. Hierzu verwendete ich MRT-Aufnahmen der Gehirne verschiedener Personen, mithilfe derer ich unterschiedliche Modelle trainierte und vergleichen konnte. Als KI-Architekturen wird ein CNN-Transformer verwendet, der mit unterschiedlich großen ResNets und EfficientNets verglichen wird. Der CNN-Transformer performt bei der Klassifizierung am besten, da er auf Lokalität und Globalität spezialisiert ist. <br>
Durch Methoden der Erklärbarkeit kann zudem die Entscheidung des Modells interpretiert und nachvollzogen werden. </em></span>

__English__ <br>
<span><em>
Alzheimer's is an increasingly common neurodegenerative disease. In its course, the brain's ability to store information decreases and the ability to speak suffers. Early detection can decisively slow down the progression of the disease, creating a better situation for those affected and their relatives.<br>
Here in this project, I want to use AI-based approaches to improve the diagnosis of Alzheimer's in all stages of the disease. To do this, I used MRI images of the brains of different people, with the help of which I was able to train and compare different models. The AI architectures used are a CNN-Transformer, which is compared with ResNets and EfficientNets of different sizes. The CNN transformer performs best in the classification because it is specialised in locality and globality. In addition, the decision of the model can be interpreted and understood through methods of Explainability. 
The model and the Explainability methods are combined in an app to help doctors diagnose Alzheimer's patients. <br>
I am also trying to find out more about the gene mechanisms in Alzheimer's disease by clustering cells and genes using Autoencoders and Graph Neural Networks to show the difference in cells from healthy patients to cells from patients with Alzheimer's disease.</em></span>

<br>
<h3>Links to the Reports:</h3>
<hr>
Report zu der Hyperparameter Optimierung des CNN-Transformer
<br>

[Weights&Biases-Reports](https://api.wandb.ai/links/nbennewiz/j8svnd3w "")
<br>

<br>
<h3>Verwenden der App</h3>
<hr>
<br>
<p>1. Python in einem Terminal öffnen</p>
<p>2. Installieren der Requirements:</p>

```python
pip install -r requirements.txt
```

<p>3. Starten der App:</p>

```python
python app.py
```

<p>4. Aufrufen des folgenden Links in einem Browser.</p>

[http://127.0.0.1:8050](http://127.0.0.1:8050 "") 

<br>
<h3>Funktionen</h3>
<hr>
<br>
<h4>Labeled MRT-Regions</h4>

![MRT-Regions](/MRI-Regions-Viewer/assets/ScreenshotMRT-Regions.png)
<h4>Overlay interactive Alzheimer-Scan</h4>

![MRT-Regions](/MRI-Regions-Viewer/assets/ScreenshotHeatmap.png)
<h4>Two labeled Atlases with different depth</h4>

![MRT-Regions](/MRI-Regions-Viewer/assets/ScreenshotLabeledAtlas.png)
<h4>View MRI-Slices by changing the axial coordinate</h4>

![MRT-Regions](/MRI-Regions-Viewer/assets/ScreenshotHumanMRT.png)

<br>
<h3>Upgraded AutoClass</h3>
<hr>

<span><em>
</em></span>


<h4>Normal Clustering</h4>

![NormalClustering](/Genexpressions-Beziehungen/assets/NormalClustering.png)
<h4>Upgraded AutoClass Clustering</h4>

![UpgradedAutoClassClustering](/Genexpressions-Beziehungen/assets/AutoClass_comparision.png)
<h4>Upgraded Disease Clustering</h4>

![UpgradedAutoClassClustering](/Genexpressions-Beziehungen/assets/AutoClass_DC_comparision.png)
<h4>Distribution differences with activation functions</h4>


|        scRNA-seq        | normal | ReLU 30| ReLU  10  | ReLU 100 |  Softplus |
| -------------- | ------ | --------- | ---- | ---- | ----- |
| Non-Zero-Count |    5.575.264    | 8.133.753 | 1.577.998 |  7.585.000 |  36.027.458     |
| Minimum        |     0.11   | 2e-07   | 0.0003 |   0.006   |  1e-45     |
| Maximum               |   24.64   |      2.74 | 3.14 |   2.78   |     2.58     |


![DistributionDifferences](/Genexpressions-Beziehungen/assets/Distributions_with_activations.png)
<br>
<br>
