

def visulizeGen(gen,d):
    gen.on_epoch_end()
    for i in range(gen.__len__()):
        Xs, Ys = gen.__getitem__(i)
        x = Xs[0]
        y = Ys[0]
        y = d.decodeImg2Mask(y)
        x = x.astype(np.uint8)
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        y = cv.cvtColor(y, cv.COLOR_BGR2RGB)
        cv.imshow("test_x", x)
        cv.imshow("test_y", y)
        cv.waitKey(0)


if __name__ == "__main__":
    import os
    import sys

    import Generators
    import Models
    import cv2 as cv
    import numpy as np
    from tensorflow import keras as k
    import pickle
    from pathlib import Path
    import tensorflow as tf

    # jobs = 6  # number of cores
    # config = tf.ConfigProto(intra_op_parallelism_threads=jobs,
    #                         inter_op_parallelism_threads=jobs,
    #                         allow_soft_placement=True,
    #                         device_count={'CPU': jobs})
    # tf.config.threading.set_inter_op_parallelism_threads(2)
    # tf.config.threading.set_intra_op_parallelism_threads(jobs)
    #tf.config.set_soft_device_placement(True)

    # session = tf.Session(config=)
    # k.set_session(session)


    modelSaveFolderName = "Synthetic_Dataset_VGG_CAT_2"
    pwd = os.path.dirname(os.path.abspath(sys.argv[0]))
    outPutFolderPath = os.path.join(pwd,"thesisModels",modelSaveFolderName)
    Path(outPutFolderPath).mkdir(parents=True, exist_ok=True)

    checkPointOutPutFolderPath = os.path.join(pwd, "thesisModels", modelSaveFolderName,modelSaveFolderName+"CP")
    Path(checkPointOutPutFolderPath).mkdir(parents=True, exist_ok=True)

    folderPath = os.path.join("E:","UnitySegOutPut","generatedData1")
    img_shape = (480,480,3)
    num_cat = 6
    batchSize = 4
    max_epoch = 150


    dataset = Generators.SyntheticDataSet(folderPath,"x","y",img_shape,num_cat)

    trainGen = Generators.CategoricalSyntheticGenerator(dataset,"train",batchSize=batchSize)
    valGen = Generators.CategoricalSyntheticGenerator(dataset,"val",batchSize=batchSize)
    testGen = Generators.CategoricalSyntheticGenerator(dataset,"test",batchSize=batchSize)
    #visulizeGen(trainGen, dataset)

    vggMod = Models.VGG_UNET(img_shape,num_cat,batchSize,isCategorical=True,doBatchNorm=False,hiddenDecoderActivation="relu",outputActivation="softmax")

    optimizer = k.optimizers.Adam(learning_rate=1e-4)
    loss = k.losses.CategoricalCrossentropy()
    vggMod.model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'],)
    print(vggMod.model.summary())
    k.utils.plot_model(vggMod.model, os.path.join(outPutFolderPath,modelSaveFolderName+".png"), show_shapes=True)

    es = k.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=7)
    rlronp = k.callbacks.ReduceLROnPlateau(monitor='loss', factor=.5, patience=3)
    cp = k.callbacks.ModelCheckpoint(filepath=checkPointOutPutFolderPath,verbose=0,monitor="val_loss",save_best_only=True,save_weights_only=False,mode="auto",save_freq="epoch",)
    cbs = [es, rlronp,cp]

    history = vggMod.model.fit(trainGen,
                        epochs=max_epoch,
                        batch_size=batchSize,
                        validation_data=valGen,
                        shuffle=True,
                        callbacks=cbs,
                        workers=16,
                        max_queue_size=64,
                        validation_freq=1,
                        )

    modelLoc = os.path.join(outPutFolderPath, 'model')
    Path(modelLoc).mkdir(parents=True, exist_ok=True)

    histLoc = os.path.join(outPutFolderPath, 'history')
    Path(histLoc).mkdir(parents=True, exist_ok=True)

    vggMod.model.save(modelLoc)
    pickle.dump(history.history, open(os.path.join(histLoc,'historym.pkl'), 'wb'))

    model = k.models.load_model(modelLoc)
    predDir = os.path.join(pwd,"X_Train","data")
    for img in os.listdir(predDir):
        toPred = os.path.join(predDir,img)
        x = cv.imread(toPred)
        x = cv.resize(x,model.input.shape[-3:-1])
        x = np.expand_dims(x,axis=0)
        y = model.predict(x)

        x = np.squeeze(x)
        y = np.squeeze(y)

        y = dataset.decodeImg2Mask(y)
        cv.imshow("x",x.astype(np.uint8))
        cv.imshow("y",cv.cvtColor(y.astype(np.uint8),cv.COLOR_RGB2BGR))
        cv.waitKey(0)