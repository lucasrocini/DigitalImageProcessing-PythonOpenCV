# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:24:19 2020

@author: lucas Rocini
"""

#modelo sequencial
from keras.models import Sequential
#camada de convolução, pooling, flatten e rede neural densa
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
#normalização das características e acelerar o desempenho da rede neural
from keras.layers.normalization import BatchNormalization
#gerar imagens adicionais
from keras.preprocessing.image import ImageDataGenerator
#numpy
import numpy as np
#trabalhar com imagens (para analisar)
from keras.preprocessing import image 

"""       
#######################
# CONVOLUÇÃO E PROCESSAMENTO DE IMAGEM
#######################
"""

#definição da rede neural
model = Sequential()

#OPERADOR DE CONVOLUÇÃO
#adicionar primeira camada de convolução
#32 = número de feature maps
# (3,3) = filtro da convolução
#input_shape = converte a dimensão da imagem, e canais RBG
#relu = função de ativação, retira os valores negativos, partes mais escuras da imagem
model.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'))

#NORMALIZATION
#pega o mapa de características gerado pelo kernel, e normaliza para entre 0 e 1
#acelera processamento
model.add(BatchNormalization())

#POOLING
#camada de pooling, matriz de 4px para pegar o maior valor (características mais importantes)
model.add(MaxPooling2D(pool_size = (2,2)))

#2ª camada de convolução
model.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))

#FLATTENING
#flatten = transforma a matriz em um vetor para passar como entrada da rede neural
model.add(Flatten())


"""       
#######################
# REDE NEURAL
#######################
"""
#cria primeira camada da rede neural
#units = quantidade de neurônios
model.add(Dense(units = 128, activation='relu')) 

#dropout
#0.2 = vai zerar 20% das entradas
model.add(Dropout(0.2))

#camada oculta
model.add(Dense(units = 128, activation='relu')) 
model.add(Dropout(0.2))

#camada de saída
#1 = uma saída somente, pois a classificação é binária (gato OU cachorro)
model.add(Dense(units = 1, activation='sigmoid')) 


model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

"""       
#######################
# DATA AUGMENTATION
#######################
"""

#rescale = normaliza os dados
#rotation_range = o grau que será feita uma rotação na imagem
#horizontal_flip = realizar giros horizontais nas imagens
#shear_range = muda os pixels para outra direção
#height_shift_range = realiza a faixa de mudança de altura
augmentation_imagens_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)

#rescale feito para que os dados de teste estejam na mesma escala
augmentation_imagens_teste = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 7,
                                   horizontal_flip = True,
                                   shear_range = 0.2,
                                   height_shift_range = 0.07,
                                   zoom_range = 0.2)


#
#carrega imagens do diretório e gera imagens com o data augmentation
#target_size = escalona as imagens para o tamanho
#class_mode = quantidade de classes à serem lidas
dataset_imagens_treinamento = augmentation_imagens_treinamento.flow_from_directory('dataset/training_set',
                                                                                   target_size = (64,64),
                                                                                   batch_size = 32,
                                                                                   class_mode = 'binary'
                                                                                   )

dataset_imagens_teste = augmentation_imagens_teste.flow_from_directory('dataset/test_set',
                                                                       target_size = (64,64),
                                                                       batch_size = 32,
                                                                       class_mode = 'binary'
                                                                      )

"""       
#######################
# TREINAMENTO
#######################
"""

#steps_per_epoch = quantidade de imagens à serem utilizadas, quanto maior melhor
#recomenda-se usar a quantidade de imagens e dividir pelo batch_size, para nao dar overflow
model.fit_generator(dataset_imagens_treinamento, 
                     steps_per_epoch= 320/32,
                     epochs = 20,
                     validation_data = dataset_imagens_teste,
                     validation_steps = 110/32 
                    )


"""
###########################                                           
# PREVISÃO INDIVIDUAL
###########################
"""

#carrega imagem
imagem_teste = image.load_img('dataset/test_set/PotesGarrafas/PoteGarrafa10000.jpg',  target_size=(64,64))

#transforma imagem em array
imagem_teste = image.img_to_array(imagem_teste)

#normalizacao
imagem_teste /= 255

#expande as dimensões do vetor
#insere
imagem_teste = np.expand_dims(imagem_teste, axis = 0)

previsao = model.predict(imagem_teste)

print("Previsão:",previsao)

print(previsao*100,"% Maquinario")
print((100-(previsao*100)),"% PotesGarrafas")

if (previsao > 0.5):
    print("Provavelmente Maquinario")
else:
    print ("Provavelmente PotesGarrafas")
    

#exibe as classes e os índices utilizados
#print(dataset_imagens_treinamento.class_indices)



