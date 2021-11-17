import os

if __name__ == '__main__':
    path1 = './Textos/Experimento2/rec.sport.hockey/'
    path2 = './Textos/Experimento2/soc.religion.christian/'
    contenido1 = os.listdir(path1)
    contenido2 = os.listdir(path2)
    comando = 'python3 TFIDFViewer.py --index exp2 --files '
    for i in range(0,102):
        for j in range(0,102):
            x = contenido1[i]
            y = contenido2[j]
            if y != x: os.system(comando + path1 + x + ' ' + path2 + y)
