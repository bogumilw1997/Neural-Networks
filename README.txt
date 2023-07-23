Aby programy działały poprawnie należy pobrać plik owid-covid-data.csv ze strony (należy kliknąć zakładkę "Download" i wybrać "Full data"):

https://ourworldindata.org/explorers/coronavirus-data-explorer

i umieścić go w folderze /data.

W paczce znajdują się 3 pliki:
1. RNN_train.py - ten program uczy sieć na danych treningowych i sprawdza predykcje na danych testowych.
2. RNN_load.py - ten program wczytuje model z katalogu /models i na jego bazie wykonuje predykcje o określonej długości.
3. RNN_predict.py - ten program bierze wszystkie dostępne dane jako dane testowe i wykonuje predykcję o określonej długości.

W folderze /data znajduje się plik parameters.json w którym są zdefiniowane następujące parametry:
- test_range: odpowiada za wielkość obszaru testowego w programie RNN_train,
- n_inputs: ilość dni, na podstawie której sieć będzie chciała przewidzieć wartość zachorowań w dniu kolejnym (najlepiej >=14),
- neurons_number: liczba neuronów warstwy LSTM,
- l_rate: wartość learning rate przy uczeniu sieci,
- epochs_number: liczba ekpok przy trenowania sieci,
- prediction_range: ilość predykcji (dni), które chcemy wykonać (dotyczy RNN_load.py oraz RNN_predict.py),
- dates_interval: określa co ile dni ma się pojawiać data na wykresie (należy zwiększyć tę wartość jeśli daty nachodzą na siebie),
- save_name: nazwa pod którą zapisze się model przy trenowaniu (dotyczy RNN_train.py),
- load_name: nazwa modelu, który zostanie wczytany (dotyczy RNN_load.py).
