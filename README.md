# <b>Alzheimer-Erkennung durch Künstliche Intelligenz</b> - JuFo2023

<h3>Kurzfassung</h3>
<hr>

<span><em>
Alzheimer ist eine zunehmend auftretende neurodegenerative Erkrankung. In ihrem Verlauf nimmt die Fähigkeit des Gehirns Informationen zu speichern ab, die Sprachfähigkeit leidet. Durch eine frühe Erkennung kann das Fortschreiten der Erkrankung entscheidend verlangsamt werden, sodass für Betroffene und Angehörige eine bessere Situation geschaffen wird.<br>
In meinem Projekt möchte ich mittels KI-basierter Ansätze die Diagnostik von Alzheimer in allen Stadien der Erkrankung verbessern. Hierzu verwendete ich MRT-Aufnahmen der Gehirne verschiedener Personen, mithilfe derer ich unterschiedliche Modelle trainierte und vergleichen konnte. Als KI-Architekturen wird ein CNN-Transformer verwendet, der mit unterschiedlich großen ResNets und EfficientNets verglichen wird. Der CNN-Transformer performt bei der Klassifizierung am besten, da er auf Lokalität und Globalität spezialisiert ist. <br>
Durch Methoden der Erklärbarkeit kann zudem die Entscheidung des Modells interpretiert und nachvollzogen werden. </em></span>


<br>
<h3>Links zu den Reports:</h3>
<hr>
Report zu der Hyperparameter Optimierung des CNN-Transformer
<br>

[Weights&Biases-Reports](https://api.wandb.ai/links/nbennewiz/j8svnd3w "")
<br>

<br>
<h3>Upgraded AutoClass</h3>
<hr>
<h4>Normal Clustering</h4>
![NormalClustering](/Genexpressions-Beziehungen/assets/NormalClustering.png)

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
