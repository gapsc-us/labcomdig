#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Conjunto de funciones desarrolladas para el manual de Laboratorio de Comunicaciones Digitales
impartido en la Escuela Técnica Superior de Ingeniería de la Universidad de Sevilla
@author: jpayan@us.es, murillo@us.es, jaradillas@us.es
'''
import numpy as np

'''
Tema 1
'''

def Qfunct(x):
    ''' 
     y = Qfunct(x) evalúa la función Q en x.
    Donde y = 1/sqrt(2*pi) * integral desde x hasta inf de exp(-t^2/2) dt
    '''
    from scipy.special import erfc 
    from math import sqrt
    y=(1/2)*erfc(x/sqrt(2)) 
    return y


'''
Tema 2
'''
    
def LCDfmp(X):    
    ''' 
     px,x = LCDfmp(X)
    Estima la función de masa de probabilidad a partir de muestras de una
    variable aleatoria discreta, en X.
    Devuelve los valores de la fmp y los puntos en los que se evalua (soporte)
    '''
    Np = np.size(X)    
    x, px = np.unique(X, return_counts=True)   
    px = px/Np
    return(px,x)    
    
def LCDfdp(X, xpuntos=128): 
    '''
     fxi,xi = LCDfdp(X, xpuntos=128)
    Estima la función densidad de probabilidad a partir de muestras de una
    variable aleatoria continua.
    Recibe los valores de la variable aleatoria de la que queremos estimar 
    la función densidad de probabilidad y un parámetro opcional. Si no se da, 
    estima la fdp en 128 intervalos. En caso contrario, este parámetro puede 
    ser bien el número de intervalos o bien los extremos de los mismos.
    
    Devuelve la estimación de la fdp en los puntos medios de los intervalos.
    '''

    fxi, xedge = np.histogram(X, bins=xpuntos, density=True)
    xi=(xedge[1:]+xedge[:-1])/2
    return fxi, xi
      
def fdpnormal(x,media=0,varianza=1):
    '''
    f = fdpnormal(x,media=0,varianza=1)
    
    Calcula el valor de la fdp de una normal con media y varianza dadas en 
    los puntos dados por x. Si media y varianza no se dan, se ponen a 0 y
    uno.
    '''
    f= 1/np.sqrt(2*np.pi*varianza)*np.exp(-(x-media)**2/(2*varianza));
    return(f)
    

'''
Tema 4
'''    
    
def transmisorpam(Bn,Eb,M,p,L):
    '''
    [Xn,Bn,An,phi,alfabetopam] = transmisorpam(Bn,M,Eb,p,L)
     
    Bn = Secuencia de dí­gitos binarios
    Eb = Energí­a media por bit transmitida en Julios
    M  = Número de sí­mbolos del código PAM
    p  = Pulso paso de baja o paso de banda 
    L  = Número de puntos a utilizar en la representación de un sí­mbolo
    
    Devuelve:
    Xn = la señal de información (discreta)
    Bn = La secuencia de dí­gitos binarios realmente transmitidos
    An = La secuencia de niveles de amplitud transmitidos
    phi = Pulso básico real normalizado (energí­a unidad)
    alfabetopam = Los niveles de amplitud asociados a cada sí­mbolo
    '''    
    
    # Se obtienen en primer lugar los niveles asociado a cada sí­mbolo ¿Cuántos bits hay en cada sí­mbolo?

    k = int(np.ceil(np.log2(M)))
    
    # Se Ajusta M a una potencia de dos
    M = 2**(k)
    
    # El alfabeto [Ver la ecuación (4.21)] 
    alfabetopam = np.sqrt(3*Eb*np.log2(M)/(M**2-1))*(2*(np.arange(M))-M+1)
    
    # Si Bn no tiene una longitud múltiplo de k, se completa con ceros
    Nb = len(Bn)  # Número de bits a transmitir, actualizado
    Bn = np.r_[Bn,np.zeros(int(k*np.ceil(Nb/k)-Nb)).astype(int)] #
    Nb = len(Bn)  # Número de bits a transmitir tras la corrección
    Ns = Nb//k        # Número de sí­mbolos a transmitir
    
    # La secuencia generada
    if M>2:
        An = alfabetopam[gray2de(np.reshape(Bn,[Ns,k]))]
    else:
        An = alfabetopam[Bn]
    
    # Comprobación de las longitudes y otros datos del pulso suministrado para 
    # hacer que el número de muestras del mismo sea efectivamente L
    Ls = len(p)
    if Ls<L:
        p = np.r_[p, np.zeros(L-Ls)]
    elif Ls>L:
        print('La duración del pulso se trunca a {} muestras'.format(str(L)))
        p = p[:L] #Debe modificarse si se quiere un pulso de más de L muestras
    
    # Se normaliza la energí­a del pulso para obtener la base del sistema
    phi = p / np.sqrt(p@p) 
       
    # Obtención del tren de pulsos
    Xn = np.kron(An,phi) #Debe modificarse si se quiere un pulso de más de L muestras
    
    return [Xn,Bn,An,phi,alfabetopam]
      
'''
Tema 5
'''
def transmisorpsk(Bn,Eb,M,p1,p2,L): 
   '''
   [Xn,Bn,An,phi1,phi2,alfabeto] = transmisorpsk(Bn,Eb,M,p1,p2,L)
   
   Entradas:
   Bn = Secuencia de dígitos binarios
   Eb = Energía media por bit transmitida en Julios
   M = Número de símbolos del código PSK
   p1 = Pulso real de la componente en fase
   p2 = Pulso real de la componente en cuadratura
   L = Número de puntos que se utilizará en la representación de un símbolo
   
   Devuelve:
   Xn = la señal de información discreta
   Bn = La secuencia de dígitos binarios realmente transmitidos
   An = La secuencia de símbolos complejos transmitidos
   phi1 = Pulso básico real normalizado (energí­a unidad) de la componente en fase
   phi2 = Pulso básico real normalizado (energí­a unidad) de la componente en cuadratura
   alfabeto = El alfabeto utilizado asociado a cada símbolo transmitido
   '''
   #Definiciones
   eps = np.finfo(float).eps # por debajo de este valor se considera cero
    
   # Comprobación de las longitudes y otros datos de los pulsos de llamada
   Ls1 = len(p1)
   Ls2 = len(p2)
   if Ls1==0 or Ls2==0:
       print('No es posible realizar la transmisión') 
       return
   
   if Ls1<L:
       p1 = np.r_[p1, np.zeros(int(L-Ls1))]
   elif Ls1>L:
       p1 = p1[:L]
       print('La duración del pulso se trunca a {} muestras'.format(str(L)))
   if Ls2<L:
       p2 = np.r_[p2, np.zeros(int(L-Ls2))]
   elif Ls2>L:
       p2 = p2[:L]
       print('La duración del pulso se trunca a {} muestras'.format(str(L)))       
       
   # Se comprueba la ortogonalidad
   if abs(p1@p2) >= 1e0*eps*10:
       print('No es posible realizar la transmisión') 
       return
       
   # Se normalizan las energías de los pulsos
   phi1 = p1 / np.sqrt(p1@p1) 
   phi2 = p2 / np.sqrt(p2@p2)  
   # Se genera el pulso complejo [Ver la ecuación (5.25)]
   phi = phi1 - 1j*phi2
   
   # Obtención de los niveles asociados a cada símbolo: alfabeto
   # Número de bits por símbolo
   k = int(np.ceil(np.log2(M)))
   # Se ajusta M a una potencia de dos
   M = 2**(k)
   # El alfabeto [Ver la ecuación (5.24)]
   alfabeto = np.sqrt(Eb*k)*np.exp(1j*2*np.pi*np.arange(M)/M) #empieza en 0
   
   # Si la longitud de Bn no es múltiplo de k, se completa con ceros
   Nb = len(Bn) 
   Bn = np.r_[Bn,np.zeros(int(k*np.ceil(Nb/k)-Nb)).astype(int)] #
   
   # Número de bits y símbolos que vamos a transmitir
   Nb = len(Bn) # Número de bits que vamos a transmitir tras la corrección 
   Ns = int(Nb/k)

   # La secuencia generada
   if M>2:
        An = alfabeto[gray2de(np.reshape(Bn,[Ns,k]))]
   else:
       An = alfabeto[Bn]
    
   # Obtención del tren de pulsos
   Xn = np.real(np.kron(An, phi))
   return [Xn, Bn, An, phi1, phi2, alfabeto]         

def transmisorqam(Bn, Eb, M1, M2, p1, p2, L): 
    '''
    [Xn,BnI,BnQ,AnI,AnQ,AI,AQ,phi1,phi2] = transmisorqam(Bn, Eb, M1, M2,p1,p2,L)
    
    Entradas:
        Bn = Secuencia de dígitos binarios
        Eb = Energía media por bit transmitida en Julios
        M1 = Nº de símbolos de la componente en fase
        M2 = Nº de símbolos de la componente en cuadratura
        p1 = Pulso real de la componente en fase
        p2 = Pulso real de la componente en cuadratura
        L = Nº de puntos que vamos a utilizar en la representación un símbolo
        
    Devuelve:
        Xn = la señal de información digital
        BnI = Bits transmitidos por la componente en fase
        BnQ = Bits transmitidos por la componente en cuadratura
        AnI = Niveles de amplitud transmitidos por la componente en fase
        AnQ = Niveles de amplitud transmitidos por la componente en cuadratura 
        AI = Niveles de amplitud usados en la componente en fase
        AQ = Niveles de amplitud usados en la componente en cuadratura
        phi1 = Pulso básico normalizado (energí­a unidad) de la componente en fase
        phi2 = Pulso básico normalizado (energí­a unidad) de la componente en cuadratura 
    '''
    
    #Definiciones
    eps = np.finfo(float).eps # por debajo de este valor se considera cero
    
    #Comprobación de las longitudes y otros datos de los pulsos básicos
    Ls1 = len(p1)
    Ls2 = len(p2)
    if Ls1==0 or Ls2==0:
        print('Pulsos de longitud 0, no es posible realizar la transmisión') 
        return
    if Ls1<L:
        p1 = np.r_[p1, np.zeros(int(L-Ls1))]
    elif Ls1>L:
        p1 = p1[:L]
        print('La duración del pulso se trunca a {} muestras'.format(str(L)))
    if Ls2<L:
        p2 = np.r_[p2, np.zeros(int(L-Ls2))]
    elif Ls1>L:
        p2 = p2[:L]
        print('La duración del pulso se trunca a {} muestras'.format(str(L)))       

    #Normalicemos las energías de los pulsos
    phi1 = 1/np.sqrt(p1@p1)*p1
    phi2 = 1/np.sqrt(p2@p2)*p2
    
    #Comprobemos la ortogonalidad
    if np.abs(phi1@phi2) >= 1e0*eps*10:
        print('Bases no ortogonales, no es posible realizar la transmisión') 
        return
    
    #Ajuste de los parámetros   
    k1 = int(np.ceil(np.log2(M1))) #Número de bits de la componente en fase
    M1 = 2**(k1) #Valor de M1 tras la corrección
    k2 = int(np.ceil(np.log2(M2))) #Número de bist de la componente en cuadratura M2 = 2**(k2) #Valor de M2 tras la corrección
    k = k1 + k2 #Número de bits en cada símbolo QAM
    Nb = len(Bn)
    Bn = np.r_[Bn,np.zeros(int(k*np.ceil(Nb/k)-Nb)).astype(int)]    
    
    #Obtención de la mitad de la distancia mínima entre símbolos para Eb dada
    A = np.sqrt(3*Eb*np.log2(M1*M2)/(M1**2+M2**2-2))
    
    #El alfabeto con los niveles
    AI = A * (2*np.arange(M1)-M1+1)
    AQ = A * (2*np.arange(M2)-M2+1)
    
    #Separación de la secuencia de bits en las secuencias de las componentes en fase y cuadratura
    BnI,BnQ = split(Bn,M1,M2)
    NbI = len(BnI)
    NbQ = len(BnQ)
    
    #Obtención de la secuencia de símbolos de las componentes en fase y cuadratura
    if M1>2:
        AnI = AI[gray2de(np.reshape(BnI,[int(NbI/k1),k1]))] 
    else:
        AnI = AI[BnI]
    if M2>2:
        AnQ = AQ[gray2de(np.reshape(BnQ,[int(NbQ/k2),k2]))]
    else:
        AnQ = AQ[BnQ]
        
    #Las componentes en fase, cuadratura y total de la señal discreta a transmitir
    XnI = np.kron(AnI, phi1);
    XnQ = np.kron(AnQ, phi2);
    Xn = XnI+XnQ;
    return Xn, BnI, BnQ, AnI, AnQ, AI, AQ, phi1, phi2 
    
'''
Tema 8
'''
def transmisorppm(Bn, Eb, g, L, M):
    '''
    [Xn,Nb,M,phi] = transmisorppm(Bn, Eb, g, L, M)
    
    Genera una señal PPM de acuerdo con los parámetros suministrados.
     Bn = Secuencia de dí­gitos binarios.
     Eb = Energia media transmitida en Julios.
     p  = Un vector que representa el pulso básico de la señal PPM que se desea generar. El  resto de los pulsos se obtienen obtiene desplazando el dado.
     L = número de puntos que vamos a utilizar para transmitir el pulso básico dentro de un sí­mbolo.
     M = número de sí­mbolos del código PPM.
    
    Devuelve:
     Xn = la señal de información digital (discreta).
     Nb = número real de bits transmitidos después de la posible corrección.
     M = número total de sí­mbolos.
     phi=una matriz en la que cada fila representa el pulso normalizado que se utiliza para transmitir cada sí­mbolo; esto es, cada fila representa un elemento de la base del espacio de señales.
    '''
    
    # Ajustemos el valor de M a una potencia de 2
    k=int(np.ceil(np.log2(M)))
    M=2**k
    Es=k*Eb    # Energí­a del sí­mbolo
    
    # Ajustemos la longitud del pulso básico.
    Ng=len(g)
    if Ng<L:
        p=np.pad(g , L-Ng, 'constant')
    else:
        p=p[:L]
    
    # Normalicemos su energí­a
    p=(1/np.sqrt(np.sum(p*p)))*p   
    
    # Ajustemos los bits a transmitir a un número entero de sí­mbolos: Si la secuencia Bn no tiene una longitud múltiplo de k, se completa con ceros
    Bn=np.pad(Bn, [0,int(k*np.ceil(len(Bn)/k)-len(Bn))], 'constant')
    
    Nb=len(Bn)  # Número de bits que vamos a transmitir
    Ns=int(Nb/k)     # Número de sí­mbolos 
    
    # Obtengamos la secuencia de M valores diferentes correspondientes a los sí­mbolos a transmitir
    An=np.dot(np.reshape(Bn,[Ns,k]), 2**np.flip(np.arange(k)))
    
    # Construyamos la matriz formada por los vectores de la base.
    phi=np.zeros([M,L*M])
    for kk in range(M):
        phi[kk,kk*L:kk*L+L]=p

    # Obtención del tren de pulsos conjunto
    Xn = list(map(lambda Ai: phi[Ai,:], An))
    Xn = np.sqrt(Es)*np.concatenate(Xn)
    
    return [Xn,Bn,Nb,M,phi]
 
'''
Otras funciones de carácter general
'''
def gray2de(b): 
    ''' 
     d = gray2de(b) Convierte cada fila de la matriz formada por dígitos binarios b
       en un vector columna, d, de los valores decimales correspondientes.
    '''
    c = np.zeros_like(b)
    c[:,0] = b[:,0]
    for i in range(1,np.shape(b)[1]):
        c[:,i] = np.logical_xor(c[:,i-1], b[:,i])
    # Convierte los bits menos significativos en los más significativos
    c = np.fliplr(c)

    #Comprueba un caso especial.
    [n,m] = np.shape(c) 
    if np.min([m,n]) < 1:
        d = []
        return
    d = np.dot(c, 2**np.arange(m))
    return d

def de2gray(d,n):
    """
    Convierte un número decimal en un vector binario de longitud n
    Versión adaptada de una función de Mathworks
    """ 
    from sympy.combinatorics.graycode import GrayCode
    
    gray_list_str = list(GrayCode(n).generate_gray())
    gray_list = list(map(lambda ind: np.array(list(gray_list_str[ind]), dtype=np.int), range(len(gray_list_str))))
    g = list(map(lambda ind: gray_list[int(d[ind])], range(0,len(d))))

    return np.array(g)
    
def split(Bn, M1, M2):
    '''
    Entradas
        Bn = una secuencia de símbolos binarios
        M1 = nº de símbolos de la componente en fase
        M2 = nº de símbolos de la componente en cuadratura    
    Salida     
        Bn = una secuencia de símbolos binarios
        M1 = nº de símbolos de la componente en fase
        M2 = nº de símbolos de la componente en cuadratura
    '''
    
    k1 = int(np.log2(M1))
    k2 = int(np.log2(M2))
    k = k1 + k2
    
    #Longitud de la secuencia
    Nb = len(Bn)
    
    #Una matriz con Ns=Nb/k filas formadas por los k bits
    W = np.reshape(Bn,[int(Nb/k),k])
    
    #Extrae la submatriz formada por los k1 primeros bits y pone una fila tras otra
    BnI = np.reshape(W[:,:k1],[k1*int(Nb/k)])
    
    #Extrae la submatriz formada por los k2 bits restantes y pone una fila tras otra
    BnQ = np.reshape(W[:,k1:],[k2*int(Nb/k)]) 
    
    return BnI, BnQ

def simbolobit(An,alfabeto):
    '''
    Bn = simbolobit(An, alfabeto)
    An       = secuencia de sí­mbolos pertenecientes al alfabeto
    alfabeto = tabla con los sí­mbolos utilizados en la transmisión 
    Bn       = una secuencia de bit, considerando que los sí­mbolos se habí­an
    generado siguiendo una codificación de Gray
    '''
    
    from labcomdig import de2gray
    
    # ¿Cuántos bits hay en cada sí­mbolo?
    k = np.log2(len(alfabeto))
    
    if k>1:
        distancia = abs(alfabeto[0]-alfabeto[1])
        indices   = np.round((An-alfabeto[0])/distancia)
        Bn        = np.reshape(de2gray(indices,k),[int(k*len(An))])
    else:
        Bn = ((An/max(alfabeto))+1)/2
    
    return Bn

def detecta(rn, alfabeto):
    '''
    detecta(rn, alfabeto)
    rn       = una secuencia de sí­mbolos más ruido
    alfabeto = tabla con los niveles de amplitud/sí­mbolos  
    
    Genera:
    An = una secuencia de sí­mbolos pertenecientes al alfabeto de acuerdo con
    una regla de distancia euclidiana mí­nima (mí­nima distancia)
    '''
    
    # Longitud de la secuencia                
    N = len(rn)
    
    # Inicializa
    An = np.zeros(N)
    
    for i in range(N):
        ind = np.where(abs(rn[i]-alfabeto) == np.amin(abs(rn[i]-alfabeto)))
        An[i] = alfabeto[ind]
    
    return An

def detectaSBF(rn,alfabeto):
    '''
    detectaSBF(rn,alfabeto)
    rn       = una secuencia de sí­mbolos más ruido
    alfabeto = tabla con los niveles de amplitud/sí­mbolos  
    
    Genera:
    An = una secuencia de sí­mbolos pertenecientes al alfabeto de acuerdo con
    una regla de distancia euclidiana mí­nima (mí­nima distancia)
    '''

    # Obtiene el í­ndice respecto al alfabeto
    ind = map(lambda i: np.argmin(abs(rn[i] - alfabeto)), range(0,len(rn)))
    
    # Genera la secuencia de niveles detectados
    An = np.array(list(map(lambda i: alfabeto[i], ind)))
    
    return An
    
# def split(Bn, M1, M2):
#     '''
#     Bn = una secuencia de símbolos binarios
#     M1 = nº de sí­mbolos de la componente en fase
#     M2 = nº de sí­mbolos de la componente en cuadratura
    
#     Devuelve:
#     BnI = La secuencia de sí­mbolos binarios de la componente en fase
#     BnQ = La secuencia de sí­mbolos binarios de la componente en cuadratura
#     '''    
#     k1=int(np.log2(M1))
#     k2=int(np.log2(M2))
#     k=k1+k2
    
#     #Longitud de la secuencia
#     Nb=len(Bn)
    
#     #Una matriz con Nb/k filas formadas por los k1 bits más los k2 bits
#     W=np.reshape(Bn,[int(Nb/k),k]) 
    
#     #Extrae la submatriz formada por los k1 primeros bits y pone una fila tras otra
#     BnI=np.reshape(W[:,:k1],[k1*int(Nb/k)])
    
#     #Extrae la submatriz formada por los k2 bits restantes y pone una fila tras otra
#     BnQ=np.reshape(W[:,k1:],[k2*int(Nb/k)])
               
#     return BnI, BnQ
    
def une(BndetectadoI,BndetectadoQ,M1,M2):
    '''
    [Bndetectado]=une(BndetectadoI,BndetectadoQ,M1,M2)
    BndetectadoI = una secuencia de sí­mbolos binarios correspondientes a los bits en posiciones múltiplos de k1
    BndetectadoQ = una secuencia de sí­mbolos binarios correspondientes a los bits en posiciones múltiplos de k2
    M1 = nº de sí­mbolos de la componente en fase
    M2 = nº de sí­mbolos de la componente en cuadratura
    '''
    
    # Devuelve
    # Bndetectado = Los bits entremezclados
    import numpy as np
    
    k1 = np.log2(M1)
    k2 = np.log2(M2)
    N = len(BndetectadoI)/k1
    C1 = np.reshape(BndetectadoI,[int(N),int(k1)])
    C2 = np.reshape(BndetectadoQ,[int(N),int(k2)])
    C = np.c_[C1,C2]
    Bndetectado = np.reshape(C,[int(N*(k1+k2))])
                         
    return Bndetectado                     