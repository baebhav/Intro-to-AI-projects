def trainnn(imagearray):
    #create input greyscale array
    greyscalearray=greyscaletransform(imagearray)
    #initialize weights using Xavier initialization
    weights0=np.zeros((5,9),dtype=np.float16)
    for i in range(weights0.shape[0]):
        for j in range(weights0.shape[1]):
            weights0[i][j]=gauss(0,m.sqrt(1/weights0.shape[1]))
        
    bias0=1
    bias0weight=0
    weights1=np.zeros((5,5),dtype=np.float16)
    for i in range(weights1.shape[0]):
        for j in range(weights1.shape[1]):
            weights1[i][j]=gauss(0,m.sqrt(1/weights1.shape[1]))
    bias1=1
    bias1weight=0
    weights2=np.zeros((3,5),dtype=np.float16)
    for i in range(weights2.shape[0]):
        for j in range(weights2.shape[1]):
            weights2[i][j]=gauss(0,m.sqrt(1/weights2.shape[1]))
    bias2=1
    bias2weight=0
    #train the neural network on pixel windows
    #image was padded to ensure that every pixel has a window around it
    for x in range(1,imagearray.shape[0]-1):
        for y in range(1,imagearray.shape[1]-1):
            inputlayer=np.array([])
            actualrgb=imagearray[x][y]
            outputlayer=np.zeros((1,3),dtype=np.float16)
            window=greyscalearray[x-1:x+2,y-1:y+2]
            for i in range(3):
                for j in range(3):
                    inputlayer=np.append(inputlayer,window[i][j])
            #pass forward and update activation values in nodes
            hiddenlayer0=np.zeros((1,5),dtype=np.float16)
            hiddenlayer1=np.zeros((1,5),dtype=np.float16)
            feedforward(inputlayer,hiddenlayer0,weights0,bias0,bias0weight)
            feedforward(hiddenlayer0,hiddenlayer1,weights1,bias1,bias1weight)
            feedforward(hiddenlayer1,outputlayer,weights2,bias2,bias2weight)
            #compute derivatives
            dCdR=2*(outputlayer[0][0]-actualrgb[0]) #derivative of red loss
            dCdG=2*(outputlayer[0][1]-actualrgb[1]) #derivative of green loss
            dCdB=2*(outputlayer[0][2]-actualrgb[2]) #derivative of blue loss
            dZ=lambda z:1/(1+m.exp(-z))*(1-(1/(1+m.exp(-z)))) #derivative of sigmoid function
            learningrate=0.1
            dRdZ=dZ(lineartransform(hiddenlayer1,weights2[0])+bias2*bias2weight) #derivative of red output with respect to sigmoid function
            dGdZ=dZ(lineartransform(hiddenlayer1,weights2[1])+bias2*bias2weight) #derivative of green output with respect to sigmoid function
            dBdZ=dZ(lineartransform(hiddenlayer1,weights2[2])+bias2*bias2weight) #derivative of blue output with respect to sigmoid function
            #backpropogate from output layer to hiddenlayer1 and update weights2
            #dRloss/dW=dRloss/dRoutput*dZ/dNode*dNode/dW
            #vectorized weight update
            weights2[0] -= (dCdR*dRdZ*hiddenlayer1[0])*learningrate
            weights2[1] -= (dCdG*dGdZ*hiddenlayer1[0])*learningrate
            weights2[2] -= (dCdB*dBdZ*hiddenlayer1[0])*learningrate
            #bias has activation unit of 1 so dNode/dW=1
            bias2weight -= (dCdR*dRdZ)*learningrate
            bias2weight -= (dCdG*dGdZ)*learningrate
            bias2weight -= (dCdB*dBdZ)*learningrate
            #backpropogate from hiddenlayer1 to hiddenlayer0 and update weights1
            for j in range(weights1.shape[0]):
                for k in range(weights1.shape[1]):
                    weights1[j][k] -= (dCdR*dRdZ*weights2[0][j]*dZ(lineartransform(hiddenlayer0,weights1[j])+bias1*bias1weight)*hiddenlayer0[0][k])*learningrate
                    weights1[j][k] -= (dCdG*dGdZ*weights2[1][j]*dZ(lineartransform(hiddenlayer0,weights1[j])+bias1*bias1weight)*hiddenlayer0[0][k])*learningrate
                    weights1[j][k] -= (dCdB*dBdZ*weights2[2][j]*dZ(lineartransform(hiddenlayer0,weights1[j])+bias1*bias1weight)*hiddenlayer0[0][k])*learningrate
            bias1weight -= (dCdR*dRdZ*weights2[0][j])*dZ(lineartransform(hiddenlayer0,weights1[j])+bias1*bias1weight)*learningrate
            bias1weight -= (dCdR*dRdZ*weights2[1][j])*dZ(lineartransform(hiddenlayer0,weights1[j])+bias1*bias1weight)*learningrate
            bias1weight -= (dCdR*dRdZ*weights2[1][j])*dZ(lineartransform(hiddenlayer0,weights1[j])+bias1*bias1weight)*learningrate
            #backpropogate from hiddenlayer0 to input layer and update weights0
            for j in range(weights0.shape[0]):
                for k in range(weights0.shape[1]):
                    redgradient=0
                    greengradient=0
                    bluegradient=0
                    for z in range(hiddenlayer1.shape[1]):
                        redgradient += dCdR*dRdZ*weights2[0][z]*dZ(lineartransform(hiddenlayer0,weights1[z])+bias1*bias1weight)*weights1[z][j]
                        greengradient += dCdG*dGdZ*weights2[1][z]*dZ(lineartransform(hiddenlayer0,weights1[z])+bias1*bias1weight)*weights1[z][j]
                        bluegradient += dCdB*dBdZ*weights2[2][z]*dZ(lineartransform(hiddenlayer0,weights1[z])+bias1*bias1weight)*weights1[z][j]
                    redgradient=redgradient*dZ(lineartransform(inputlayer,weights0[j])+bias0*bias0weight)*inputlayer[k]
                    greengradient=greengradient*dZ(lineartransform(inputlayer,weights0[j])+bias0*bias0weight)*inputlayer[k]
                    bluegradient=bluegradient*dZ(lineartransform(inputlayer,weights0[j])+bias0*bias0weight)*inputlayer[k]
                    
                    weights0[j][k] -= (redgradient+greengradient+bluegradient)*learningrate
    
            print(x,y)
    return [weights0,bias0weight,weights1,bias1weight,weights2,bias2weight]       
def feedforward(layer0,layer1,weights,bias,biasweight):
    for i in range(layer1.shape[1]):
        layer1[0][i]=sigmoid(lineartransform(layer0,weights[i])+bias*biasweight)
def sigmoid(activation):
    return 1/(1+m.exp(-activation))
def lineartransform(inputlayer,weights):
    return np.matmul(inputlayer,np.transpose(weights))
def predict(greyscalearray,model):
    weights0=model[0]
    bias0weight=model[1]
    weights1=model[2]
    bias1weight=model[3]
    weights2=model[4]
    bias2weight=model[5]
    
    coloredimagearray=np.zeros(greyscalearray.shape[0],greyscalearray.shape[1],3)
    for x in range(1,coloredimagearray.shape[0]-1):
        for y in range(1,coloredimagearray.shape[1]-1):
            inputlayer=np.array([])
            outputlayer=np.zeros((1,3),dtype=np.float16)
            window=greyscalearray[x-1:x+2,y-1:y+2]
            for i in range(3):
                for j in range(3):
                    inputlayer=np.append(inputlayer,window[i][j])
            hiddenlayer0=np.zeros((1,10),dtype=np.float16)
            hiddenlayer1=np.zeros((1,10),dtype=np.float16)
            feedforward(inputlayer,hiddenlayer0,weights0,bias0,bias0weight)
            feedforward(hiddenlayer0,hiddenlayer1,weights1,bias1,bias1weight)
            feedforward(hiddenlayer1,outputlayer,weights2,bias2,bias2weight)
            coloredimagearray[x][y]=np._int(outputlayer*255)
    plt.imsave("coloredimage.jpeg",coloredimagearray)
    
def preprocessing(image):
    #add padding to all four sides of image. Pixel values are the same as the neighboring pixels
    #from the original image. Adding padding allows every pixel to be at the center of a window.
    loadimage=cv2.imread(image)
    #adding padding row above
    toppadding=np.zeros((1,loadimage.shape[1],3),dtype=np.uint8)
    for i in range(loadimage.shape[1]):
        for j in range(3):
            toppadding[0][i][j]=loadimage[0][i][j]
    #adding padding row below  
    loadimage=np.concatenate((toppadding,loadimage))
    bottompadding=toppadding=np.zeros((1,loadimage.shape[1],3),dtype=np.uint8)
    for i in range(loadimage.shape[1]):
        for j in range(3):
            bottompadding[0][i][j]=loadimage[loadimage.shape[0]-1][i][j]
    loadimage=np.concatenate((loadimage,bottompadding))
    #add padding to the left
    leftpadding=np.zeros((loadimage.shape[0],1,3),dtype=np.uint8)
    for i in range(loadimage.shape[0]):
        for j in range(3):
            leftpadding[i][0][j]=loadimage[i][0][j]
    loadimage=np.concatenate((leftpadding,loadimage),axis=1)
    #add padding to the right
    rightpadding=np.zeros((loadimage.shape[0],1,3),dtype=np.uint8)
    for i in range(loadimage.shape[0]):
        for j in range(3):
            rightpadding[i][0][j]=loadimage[i][loadimage.shape[1]-1][j]
    loadimage=np.concatenate((loadimage,rightpadding),axis=1)
    #cv2 loads images using BGR format so the pixel arrays need to be flipped
    for i in range(loadimage.shape[0]):
        for j in range(loadimage.shape[1]):
            loadimage[i][j]=np.flip(loadimage[i][j])
            
    #convert rgb values to float values between 0 and 1. Neural network training is more efficient
    #with float pixel values
    floatimage=np.float16(loadimage)
    for i in range(floatimage.shape[0]):
        for j in range(floatimage.shape[1]):
            floatimage[i][j]=(floatimage[i][j])/255
    return floatimage