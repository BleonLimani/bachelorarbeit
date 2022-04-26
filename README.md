# Bachelorarbeit

Vergleich von Methoden zur Wiederherstellung von fehlenden Werten in Zeitreihen

## bar_graph

Enthält alle Balkendiagramme die aus den Dateien in pickle und pickle2 generiert wurden. Zusätzlich befindet sich dort die ipynb Datei und der Code der für das Erstellen dieser Balkendiagramme benutzt wurde.


## datasets

Im Ordner Datasets findet man einige csv Dateien wobei nur zwei von diesen für die Bachelorarbeit benutzt wurden.


## mypackage

Enthält die Python Datei mit den neu entwickelten Methoden die auf Clusterings basieren. Zusätzlich lassen sich noch weitere Funktionen finden die für die Erstellung der Dateien in pickle und pickle2 notwendig waren.

## pickle

In diesem Ordner befinden sich alle mit pickle_data_from_generated_data2_dataset.ipynb generierten Dateien. Bei diesen Dateien wurde eine feste Anzahl an zufällig gewählten Datenpunkten aus dem Datensatz mit NaN ersetzt um so fehlende Werte zu simulieren.


## pickle2

In diesem Ordner befinden sich alle mit pickle_data_from_generated_data_dataset.ipynb generierten Dateien. Bei diesen Dateien wurde eine feste Anzahl an zufällig gewählten Datenpunkten aus jeder Zeitreihe mit NaN ersetzt um so fehlende Werte zu simulieren.


## Aufbau der Dateien in pickle und pickle2

```python
pickle_avg_X_Y -> Ist eine Liste welche pro Index die durchschnittlichen Abstände der wiederhergestellten Daten des DataFrames mit dem dazugehörigen Index anzeigt.
pickle_df_X_Y -> Ist eine Liste die pro Index ein DataFrame besitzt. Jede verglichene Methode besitzt hier ihre eigene Reihe
pickle_rating_X_Y -> Ist eine Liste die pro Index das beste CLOSE-Rating und die dazugehörigen epsilon und min_samples Werte beinhaltet die für DBSCAN benutzt wurden. 
                     [0] = CLOSE-Score
                     [1] = epsilon
                     [2] = min_samples

pickle ->  X = Anzahl der fehlenden Datenpunkte pro Datensatz
           Y = Anzahl an Testdurchläufen oder anders Länge der Liste

pickle2 -> X = Anzahl der fehlenden Datenpunkte pro Zeitreihe
           Y = Anzahl an Testdurchläufen oder anders Länge der Liste
```
