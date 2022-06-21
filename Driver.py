import threading
import time

from path_plan import WaveFront
from ImageStitching import ImageStreamStitcher
from scipy import ndimage as nd


def visulizeGenCat(gen,d):
    gen.on_epoch_end()
    for i in range(gen.__len__()):
        Xs, Ys = gen.__getitem__(i)
        x = Xs[0]
        y = Ys[0]
        y = d.decodeOneHot2Mask(y)
        x = x.astype(np.uint8)
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        y = cv.cvtColor(y, cv.COLOR_BGR2RGB)
        cv.imshow("test_x", x)
        cv.imshow("test_y", y)
        cv.waitKey(0)

def visulizeGenReg(gen,d):
    gen.on_epoch_end()
    for i in range(gen.__len__()):
        Xs, Ys = gen.__getitem__(i)
        x = Xs[0]
        y = Ys[0]
        y = y*255
        x = x.astype(np.uint8)
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        #y = cv.cvtColor(y, cv.COLOR_BGR2RGB)
        y = cv.applyColorMap(y.astype('uint8'), cv.COLORMAP_HOT)
        cv.imshow("test_x", x)
        cv.imshow("test_y", y)
        cv.waitKey(0)


def trainCatDeepLab(trainGen,valGen,batchSize,img_shape,num_cat,max_epoch,outPutFolderPath,modelSaveFolderName,checkPointOutPutFolderPath,restoreFromCp = False):

    if restoreFromCp:
        model = k.models.load_model(checkPointOutPutFolderPath)
    else:
        DeepLabMod = Models.Deep_Lab_V3(img_shape,num_cat,outputActivation="softmax")

        model = DeepLabMod.model
        optimizer = k.optimizers.Adam(learning_rate=1e-4)
        loss = k.losses.CategoricalCrossentropy()
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'], )
        print(model.summary())
        k.utils.plot_model(model, os.path.join(outPutFolderPath, modelSaveFolderName + ".png"), show_shapes=True)

    es = k.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=7)
    rlronp = k.callbacks.ReduceLROnPlateau(monitor='loss', factor=.5, patience=3)
    cp = k.callbacks.ModelCheckpoint(filepath=checkPointOutPutFolderPath, verbose=0, monitor="val_loss",
                                     save_best_only=True, save_weights_only=False, mode="auto", save_freq="epoch", )
    cbs = [es, rlronp, cp]



    history = model.fit(trainGen,
                               epochs=max_epoch,
                               batch_size=batchSize,
                               validation_data=valGen,
                               shuffle=True,
                               callbacks=cbs,
                               workers=16,
                               max_queue_size=64,
                               validation_freq=1,
                               )

    return model, history


def trainRegDeepLab(trainGen, valGen, batchSize, img_shape, max_epoch, outPutFolderPath, modelSaveFolderName,
                checkPointOutPutFolderPath,restoreFromCp = False):

    if restoreFromCp:
        model = k.models.load_model(checkPointOutPutFolderPath)
    else:
        DeepLabMod = Models.Deep_Lab_V3(img_shape, num_classes=1,outputActivation="max_relu")

        model = DeepLabMod.model
        optimizer = k.optimizers.Adam(learning_rate=1e-4)
        loss = k.losses.MeanAbsoluteError()
        model.compile(loss=loss, optimizer=optimizer, metrics=['mean_absolute_percentage_error'], )
        print(model.summary())
        k.utils.plot_model(model, os.path.join(outPutFolderPath, modelSaveFolderName + ".png"), show_shapes=True)

    es = k.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=7)
    rlronp = k.callbacks.ReduceLROnPlateau(monitor='loss', factor=.5, patience=3)
    cp = k.callbacks.ModelCheckpoint(filepath=checkPointOutPutFolderPath, verbose=0, monitor="val_loss",
                                     save_best_only=True, save_weights_only=False, mode="auto", save_freq="epoch", )
    cbs = [es, rlronp, cp]

    history = model.fit(trainGen,
                        epochs=max_epoch,
                        batch_size=batchSize,
                        validation_data=valGen,
                        shuffle=True,
                        callbacks=cbs,
                        workers=16,
                        max_queue_size=64,
                        validation_freq=1,
                        )

    return model, history



def trainCatVGG(trainGen,valGen,batchSize,img_shape,num_cat,max_epoch,outPutFolderPath,modelSaveFolderName,checkPointOutPutFolderPath,restoreFromCp = False):

    if restoreFromCp:
        model = k.models.load_model(checkPointOutPutFolderPath)
    else:
        vggMod = Models.VGG_UNET(img_shape, num_cat, batchSize, isCategorical=True, doBatchNorm=False,
                                 hiddenDecoderActivation="relu", outputActivation="softmax")

        model = vggMod.model
        optimizer = k.optimizers.Adam(learning_rate=1e-4)
        loss = k.losses.CategoricalCrossentropy()
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'], )
        print(model.summary())
        k.utils.plot_model(model, os.path.join(outPutFolderPath, modelSaveFolderName + ".png"), show_shapes=True)

    es = k.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=7)
    rlronp = k.callbacks.ReduceLROnPlateau(monitor='loss', factor=.5, patience=3)
    cp = k.callbacks.ModelCheckpoint(filepath=checkPointOutPutFolderPath, verbose=0, monitor="val_loss",
                                     save_best_only=True, save_weights_only=False, mode="auto", save_freq="epoch", )
    cbs = [es, rlronp, cp]

    history = model.fit(trainGen,
                               epochs=max_epoch,
                               batch_size=batchSize,
                               validation_data=valGen,
                               shuffle=True,
                               callbacks=cbs,
                               workers=16,
                               max_queue_size=64,
                               validation_freq=1,
                               )

    return model, history


def trainRegVGG(trainGen, valGen, batchSize, img_shape, max_epoch, outPutFolderPath, modelSaveFolderName,
                checkPointOutPutFolderPath,restoreFromCp = False):

    if restoreFromCp:
        model = k.models.load_model(checkPointOutPutFolderPath)
    else:
        vggMod = Models.VGG_UNET(img_shape, num_classes=1, batchSize=batchSize, isCategorical=True, doBatchNorm=True,
                                 hiddenDecoderActivation="relu", outputActivation="max_relu")

        model = vggMod.model
        optimizer = k.optimizers.Adam(learning_rate=1e-4)
        loss = k.losses.MeanAbsoluteError()
        model.compile(loss=loss, optimizer=optimizer, metrics=['mean_absolute_percentage_error'], )
        print(model.summary())
        k.utils.plot_model(model, os.path.join(outPutFolderPath, modelSaveFolderName + ".png"), show_shapes=True)

    es = k.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=7)
    rlronp = k.callbacks.ReduceLROnPlateau(monitor='loss', factor=.5, patience=3)
    cp = k.callbacks.ModelCheckpoint(filepath=checkPointOutPutFolderPath, verbose=0, monitor="val_loss",
                                     save_best_only=True, save_weights_only=False, mode="auto", save_freq="epoch", )
    cbs = [es, rlronp, cp]

    history = model.fit(trainGen,
                        epochs=max_epoch,
                        batch_size=batchSize,
                        validation_data=valGen,
                        shuffle=True,
                        callbacks=cbs,
                        workers=16,
                        max_queue_size=64,
                        validation_freq=1,
                        )

    return model, history


def getSyntheticTrainedModelNames(modelArch : str,categorical : bool,
                                  vggCatDefault = "Synthetic_Dataset_VGG_CAT_3",
                                  vggRegDefault = "Synthetic_Dataset_VGG_REG_1",
                                  deepLabCatDefault = "Synthetic_Dataset_DEEPLAB_CAT_1",
                                  deepLabRegDefault = "Synthetic_Dataset_DEEPLAB_REG_1"):
    modelArch = modelArch.lower()

    assert modelArch == 'vgg' or modelArch == 'deeplab', \
        'only vgg and deep lab are implemnted check args or add implemtation, Recived ' + modelArch + " for model arc"

    if modelArch == 'vgg':
        if categorical:
            modelSaveFolderName = vggCatDefault

        else:  # it is regression
            modelSaveFolderName = vggRegDefault
    elif modelArch == 'deeplab':
        if categorical:
            modelSaveFolderName = deepLabCatDefault
        else:  # it is regression
            modelSaveFolderName = deepLabRegDefault

    return modelSaveFolderName

def runTrainMode(modelArch, categorical,modelSaveFolderName,dataFolder,restoreFromCp=False):






    pwd = os.path.dirname(os.path.abspath(sys.argv[0]))
    outPutFolderPath = os.path.join(pwd, "thesisModels", modelSaveFolderName)
    Path(outPutFolderPath).mkdir(parents=True, exist_ok=True)

    checkPointOutPutFolderPath = os.path.join(pwd, "thesisModels", modelSaveFolderName, modelSaveFolderName + "CP")
    Path(checkPointOutPutFolderPath).mkdir(parents=True, exist_ok=True)


    img_shape = (480, 480, 3)
    num_cat = 6
    batchSize = 4
    max_epoch = 150

    dataset = Generators.SyntheticDataSet(dataFolder, "x", "y", img_shape, num_cat)

    ##################################################################
    ###################Model Training##########################
    ####################################################################
    if modelArch == 'vgg':
        if categorical:
            ############## train the categorical VGG model##############
            trainGen = Generators.CategoricalSyntheticGenerator(dataset, "train", batchSize=batchSize)
            valGen = Generators.CategoricalSyntheticGenerator(dataset, "val", batchSize=batchSize)
            testGen = Generators.CategoricalSyntheticGenerator(dataset, "test", batchSize=batchSize)
            #visulizeGenCat(trainGen, dataset)
            model, history = trainCatVGG(trainGen,valGen,batchSize,img_shape,num_cat,max_epoch,
                                         outPutFolderPath,modelSaveFolderName,checkPointOutPutFolderPath,
                                         restoreFromCp=restoreFromCp)

        ########### train the regression VGG model#####################
        else:
            trainGen = Generators.RegressionSyntheticGenerator(dataset, "train", batchSize=batchSize)
            valGen = Generators.RegressionSyntheticGenerator(dataset, "val", batchSize=batchSize)
            testGen = Generators.RegressionSyntheticGenerator(dataset, "test", batchSize=batchSize)
            visulizeGenReg(trainGen, dataset)
            model, history = trainRegVGG(trainGen, valGen, batchSize, img_shape, max_epoch,
                                         outPutFolderPath, modelSaveFolderName, checkPointOutPutFolderPath,
                                         restoreFromCp=restoreFromCp)

    elif modelArch == 'deeplab':
        if categorical:
            ############## train the categorical Deep Lab model##############
            trainGen = Generators.CategoricalSyntheticGenerator(dataset, "train", batchSize=batchSize)
            valGen = Generators.CategoricalSyntheticGenerator(dataset, "val", batchSize=batchSize)
            testGen = Generators.CategoricalSyntheticGenerator(dataset, "test", batchSize=batchSize)
            visulizeGenCat(valGen, dataset)
            # model, history = trainCatDeepLab(trainGen,valGen,batchSize,img_shape,num_cat,max_epoch,
            #                              outPutFolderPath,modelSaveFolderName,checkPointOutPutFolderPath,
            #                                  restoreFromCp=restoreFromCp)

        else:
            ############ train the regression DeepLab model#####################

            trainGen = Generators.RegressionSyntheticGenerator(dataset, "train", batchSize=batchSize)
            valGen = Generators.RegressionSyntheticGenerator(dataset, "val", batchSize=batchSize)
            testGen = Generators.RegressionSyntheticGenerator(dataset, "test", batchSize=batchSize)
            visulizeGenReg(trainGen, dataset)
            model, history = trainRegDeepLab(trainGen, valGen, batchSize, img_shape, max_epoch,
                                         outPutFolderPath, modelSaveFolderName, checkPointOutPutFolderPath,
                                         restoreFromCp=restoreFromCp)

    ######################################################################
    #################end Model train##########################
    ####################################################################

    ############ save the final model and history##################
    modelLoc = os.path.join(outPutFolderPath, 'model')
    Path(modelLoc).mkdir(parents=True, exist_ok=True)

    histLoc = os.path.join(outPutFolderPath, 'history')
    Path(histLoc).mkdir(parents=True, exist_ok=True)

    model.save(modelLoc)
    pickle.dump(history.history, open(os.path.join(histLoc, 'historym.pkl'), 'wb'))


def visModelOutForFloodNet(modelSaveFolderName,categorical):


    ############# visuilize prediction on the FloodNet dataset#########

    img_shape = (480, 480, 3)
    num_cat = 6
    dataset = Generators.SyntheticDataSet(None, None, None, img_shape, num_cat)

    pwd = os.path.dirname(os.path.abspath(sys.argv[0]))
    outPutFolderPath = os.path.join(pwd, "thesisModels", modelSaveFolderName)
    modelLoc = os.path.join(outPutFolderPath, 'model')
    model = k.models.load_model(modelLoc)
    predDir = os.path.join(pwd,"Unlabeled","image")
    # predDir = os.path.join(pwd,"X_Train","data")
    imgs =os.listdir(predDir)
    for img in imgs:
    # for img in [imgs[86],imgs[45]]:
        toPred = os.path.join(predDir,img)

        for shirk in [1, 2, 3, 4]:

            newSize = (img_shape[0] // shirk, img_shape[1] // shirk)
            topAndLeft = (img_shape[0] - img_shape[0] // shirk) // 2
            botAndRight = np.ceil((img_shape[0] - img_shape[0] // shirk) / 2).astype(int)

            ix = k.preprocessing.image.load_img(toPred)
            x = k.preprocessing.image.img_to_array(ix)


            x = cv.resize(x,img_shape[:2])


            cv.resize(x, newSize)
            x = cv.copyMakeBorder(cv.resize(x, newSize), topAndLeft, botAndRight, topAndLeft, botAndRight,
                                  cv.BORDER_CONSTANT,value=0)
            x = np.expand_dims(x, axis=0)

            y = model.predict(x)

            x = np.squeeze(x)
            y = np.squeeze(y)



            if categorical:

                y_pred_px = dataset.costMapFromEncoded(np.argmax(y, axis=2).astype(np.float32))
                y = dataset.decodeOneHot2Mask(y)

                y_plt =cv.cvtColor(y.astype(np.uint8),cv.COLOR_RGB2BGR)
                y_px_plt =cv.applyColorMap(dataset.scaleNormedCostMapForPlot(y_pred_px), cv.COLORMAP_HOT)

                y_plt[x==0] =0
                y_px_plt[x==0] = 0

                cv.imshow("y " + str(newSize),y_plt)
                cv.imshow("pred_px " + str(newSize),y_px_plt )

            else:
                y = dataset.scaleNormedCostMapForPlot(y)
                y_plt = cv.applyColorMap(y.astype('uint8'), cv.COLORMAP_HOT).astype(np.uint8)
                cv.imshow("y " + str(newSize), y_plt)

            x = cv.cvtColor(x.astype(np.uint8), cv.COLOR_RGB2BGR)


            cv.imshow("x " + str(newSize) , x)

        cv.waitKey(0)

def getPixelCostFromClassGT(iy,dataset):
    gt_px = iy.copy()
    foobarMask = None
    for mask_val in dataset.mask2class_encoding:
        if foobarMask is None:
            foobarMask = np.any(gt_px[:, :] != mask_val, axis=2)
        else:
            foobarMask = np.logical_and(foobarMask, np.any(gt_px[:, :] != mask_val, axis=2))

    for mask_val, enc in dataset.mask2class_encoding.items():
        gt_px[np.all(gt_px == mask_val, axis=2)] = enc

    # ideally we need to figure out how to make unity behave not shadding flat??
    indices = nd.distance_transform_edt(foobarMask, return_distances=False, return_indices=True)
    gt_px = gt_px[tuple(indices)]

    # only need one dimension
    gt_px = gt_px[:, :, 0]

    gt_px = dataset.costMapFromEncoded(gt_px)

    return gt_px



def visModelOutForSynthData(modelSaveFolderName,dataFolder,categorical):


    ########### visuilize prediction on test data set ##########
    img_shape = (480, 480, 3)
    num_cat = 6
    dataset = Generators.SyntheticDataSet(dataFolder, "x", "y", img_shape, num_cat)

    pwd = os.path.dirname(os.path.abspath(sys.argv[0]))
    outPutFolderPath = os.path.join(pwd, "thesisModels", modelSaveFolderName)
    modelLoc = os.path.join(outPutFolderPath, 'model')

    model = k.models.load_model(modelLoc)
    predDir = os.path.join(dataFolder, "x")
    test = dataset.getPartition(.15, .15)['test']
    # for img in os.listdir(predDir):
    for img in ['Hiroshima_22.png','Chantilly_162.png']:
        if img in test:

            toPred = os.path.join(predDir, img)
            ix = k.preprocessing.image.load_img(toPred)
            x = k.preprocessing.image.img_to_array(ix)
            x = cv.resize(x, img_shape[:2])
            x = np.expand_dims(x, axis=0)
            y_pred = model.predict(x)

            x = np.squeeze(x)
            y_pred = np.squeeze(y_pred)



            iy = k.preprocessing.image.load_img(os.path.join(dataFolder,'y',img))
            iy = k.preprocessing.image.img_to_array(iy)

            gt_px = getPixelCostFromClassGT(iy, dataset)



            if categorical:


                y_pred_px = dataset.costMapFromEncoded(np.argmax(y_pred, axis=2).astype(np.float32))

                y_pred = dataset.decodeOneHot2Mask(y_pred)
                cv.imshow("y_pred", cv.cvtColor(y_pred.astype(np.uint8), cv.COLOR_RGB2BGR))
                cv.imshow('y_gt',cv.cvtColor(iy.astype(np.uint8), cv.COLOR_RGB2BGR))

                cv.imshow("pred_px",cv.applyColorMap(dataset.scaleNormedCostMapForPlot(y_pred_px), cv.COLORMAP_HOT))
                cv.imshow("GT_px",cv.applyColorMap(dataset.scaleNormedCostMapForPlot(gt_px), cv.COLORMAP_HOT))
            else:
                y_pred = dataset.scaleNormedCostMapForPlot(y_pred)
                y_pred = cv.applyColorMap(y_pred.astype('uint8'), cv.COLORMAP_HOT)
                cv.imshow("y_pred", y_pred.astype(np.uint8))
                cv.imshow("GT_px", cv.applyColorMap(dataset.scaleNormedCostMapForPlot(gt_px), cv.COLORMAP_HOT))

            cv.imshow("x", cv.cvtColor(x.astype(np.uint8), cv.COLOR_RGB2BGR))
            cv.waitKey(0)


def getEvalMetricsForModel(modelSaveFolderName,dataFolder,categorical):

    batchSize = 1

    ########### visuilize prediction on test data set ##########
    img_shape = (480, 480, 3)
    num_cat = 6
    dataset = Generators.SyntheticDataSet(dataFolder, "x", "y", img_shape, num_cat)

    pwd = os.path.dirname(os.path.abspath(sys.argv[0]))
    outPutFolderPath = os.path.join(pwd, "thesisModels", modelSaveFolderName)
    modelLoc = os.path.join(outPutFolderPath, 'model')

    model = k.models.load_model(modelLoc)


    if categorical:

        testGen = Generators.CategoricalSyntheticGenerator(dataset, "test", batchSize=batchSize)
        # visulizeGenCat(testGen, dataset)

        maeList = []
        for x_batch,y_gt_batch in testGen:
            for i in range(len(x_batch)):
                x = x_batch[i]
                y_gt = y_gt_batch[i]
                x = np.expand_dims(x, axis=0)
                y_pred = model.predict(x)
                y_pred = np.squeeze(y_pred)
                y_pred = dataset.costMapFromEncoded(np.argmax(y_pred, axis=2).astype(np.float32))
                y_gt = dataset.costMapFromEncoded(np.argmax(y_gt, axis=2).astype(np.float32))

                mae = np.absolute(np.subtract(y_gt, y_pred)).mean()
                maeList.append(mae)
                # print(mae)

        mae = np.mean(maeList)
        print("Mean Average error for between GT cost map and generated Pred Cost map " + str(mae))
        loss_cross_entropy, accuracyMetric = model.evaluate(testGen,verbose=1,)#batch_size=batchSize)
        print("model loss Cross Entropy: " +str(loss_cross_entropy))
        print("Accuracy: " + str(accuracyMetric))


    else:

        testGen = Generators.RegressionSyntheticGenerator(dataset, "test", batchSize=batchSize)
        loss_mae, maePercentMetric = model.evaluate(testGen,verbose=1)
        print("model loss MAE: "+str(loss_mae))
        print("model mean average percent error : "+str(loss_mae))


def generateateNavResultForAllModelsAtLoc(imgs2ClassFolder,finalSaveName):

    img_shape = (480, 480, 3)
    num_cat = 6
    batchSize = 4
    max_epoch = 150

    calc_downSample = 8
    plottingUpSample = 1 / 2

    dataset = Generators.SyntheticDataSet(None, None, None, img_shape, num_cat)

    pwd = os.path.dirname(os.path.abspath(sys.argv[0]))




    def costPredictor(x):
        x = cv.resize(x, img_shape[:2])
        x = np.expand_dims(x, axis=0)
        y = model.predict(x)

        # x = np.squeeze(x)
        y = np.squeeze(y)

        if categorical:
            y = dataset.costMapFromEncoded(np.argmax(y, axis=2).astype(np.float32))

        return dataset.scaleNormedCostMap(y)

    folderStitcher = None
    ex, ey, sx, sy = None, None, None, None
    plotPathOn = None

    modelArchs = ['deeplab', 'vgg']
    categories = [True, False]

    pathColors = [(255,0,0),(0,255,0),(255,255,255),(0,0,255)]

    idx = 0
    for modelArch in modelArchs:
        for categorical in categories:

            modelSaveFolderName = getSyntheticTrainedModelNames(modelArch, categorical)
            outPutFolderPath = os.path.join(pwd, "thesisModels", modelSaveFolderName)
            modelLoc = os.path.join(outPutFolderPath, 'model')
            model = k.models.load_model(modelLoc)

            if folderStitcher is None:
                folderStitcher = ImageStreamStitcher(isaffine=True)

            p, cm, gtcm = folderStitcher.consumeFolder(imgs2ClassFolder, costPredictor, dataset,plot=False)

            w = WaveFront(p, cm, dataset, calc_downSample)
            hm = w.getPCMHeatMapPlot(plottingUpSample)
            if ex is None and ey is None:
                w.manualSetEndPoints(plottingUpSample,
                                     imgPlot=w.getImgAtHeatPlotSize(plottingUpSample),
                                     heatMapPlot=hm,
                                     plotName="wave")
                ex = w.ex
                ey = w.ey

            else:
                w.ex = ex
                w.ey = ey

            w.generateWaveCostMap(plottingUpSample=plottingUpSample,
                                  plotName="WaveFrontHeatMap",
                                  plot=True)

            if sx is None and sy is None:
                sx, sy = w.userDefinedPoint(plottingUpSample, imgPlot=w.getImgAtHeatPlotSize(plottingUpSample),
                                               heatMapPlot=w.getHeatPlotForWave(plottingUpSample), plotName="WaveFrontHeatMap")

            cvWaitKey  = 1
            isLast = modelArch == modelArchs[-1] and categorical == categories[-1]

            if isLast:
                cvWaitKey = 0

            if plotPathOn is None:
                plotPathOn =w.getImgAtHeatPlotSize(plottingUpSample)

            plotPathOn, cost =w.plotGlobalPath("Wave Front Global Path", plottingUpSample, sx, sy,
                                imgPlot=plotPathOn,
                                plotImeadiate=True, cvWaitKey=cvWaitKey,pathColor=pathColors[idx],
                                weightMap=gtcm)

            strModel = modelArch + " Categorical Model " if categorical else "Regresor Model "
            print("Cost for " + strModel + str(cost) +
                  " color of path " + str(pathColors[idx]))


            idx = idx+1
            del model


    cv.imwrite(finalSaveName,plotPathOn)

def fullAutoFolder(modelArch,categorical,imgs2ClassFolder,PanoramaSaveLoc = None):

    img_shape = (480, 480, 3)
    num_cat = 6
    batchSize = 4
    max_epoch = 150

    calc_downSample = 8
    plottingUpSample = 1 / 2

    dataset = Generators.SyntheticDataSet(None, None, None, img_shape, num_cat)

    pwd = os.path.dirname(os.path.abspath(sys.argv[0]))

    modelSaveFolderName = getSyntheticTrainedModelNames(modelArch, categorical)
    outPutFolderPath = os.path.join(pwd, "thesisModels", modelSaveFolderName)
    modelLoc = os.path.join(outPutFolderPath, 'model')
    model = k.models.load_model(modelLoc)

    def costPredictor(x):
        x = cv.resize(x, img_shape[:2])
        x = np.expand_dims(x, axis=0)
        y = model.predict(x)

        # x = np.squeeze(x)
        y = np.squeeze(y)

        if categorical:
            y = dataset.costMapFromEncoded(np.argmax(y, axis=2).astype(np.float32))

        return dataset.scaleNormedCostMap(y)

    folderStitcher = ImageStreamStitcher(isaffine=True)
    p, cm = folderStitcher.consumeFolder(imgs2ClassFolder, costPredictor,dataset)

    if PanoramaSaveLoc is not None:
        Path(PanoramaSaveLoc).mkdir(parents=True, exist_ok=True)

    w = WaveFront(p, cm, dataset, calc_downSample)
    w.fullWaveFrontNavEX(plottingUpSample)


def runClassificationOnFolder(categorical, modelSaveFolderName,folderPath):


    # folderPath = os.path.join("/home", "samiw", "thesis", "data", "UnitySegOutPut","generatedDataCOPY")

    img_shape = (480, 480, 3)
    num_cat = 6
    batchSize = 4
    max_epoch = 150

    dataset = Generators.SyntheticDataSet(folderPath, "x", "y", img_shape, num_cat)

    pwd = os.path.dirname(os.path.abspath(sys.argv[0]))
    outPutFolderPath = os.path.join(pwd, "thesisModels", modelSaveFolderName)
    modelLoc = os.path.join(outPutFolderPath, 'model')
    model = k.models.load_model(modelLoc)
    predDir = os.path.join(folderPath, "x")
    # test = dataset.getPartition(.15, .15)['test']
    for img in os.listdir(predDir):
        # if img in test:
        toPred = os.path.join(predDir, img)

        ix = k.preprocessing.image.load_img(toPred)
        x = k.preprocessing.image.img_to_array(ix)

        x = cv.resize(x, img_shape[:2])
        x = np.expand_dims(x, axis=0)
        y = model.predict(x)

        x = np.squeeze(x)
        y = np.squeeze(y)

        if categorical:
            yh = dataset.costMapFromEncoded(np.argmax(y, axis=2).astype(np.float32))
            with open(os.path.join(folderPath, "pred_raw", img[:-4] + '.npy'), 'wb') as f:
                np.save(f, dataset.scaleNormedCostMap(yh))
            yh = dataset.scaleNormedCostMapForPlot(yh)
            yh = cv.applyColorMap(yh.astype('uint8'), cv.COLORMAP_HOT)
            cv.imshow("yh", yh.astype(np.uint8))
            y = dataset.decodeOneHot2Mask(y)
            cv.imshow("y", cv.cvtColor(y.astype(np.uint8), cv.COLOR_RGB2BGR))
            cv.imwrite(os.path.join(folderPath, "pred_img", img), yh)
        else:
            y = dataset.scaleNormedCostMapForPlot(y)
            y = cv.applyColorMap(y.astype('uint8'), cv.COLORMAP_HOT)
            cv.imshow("y", y.astype(np.uint8))

        cv.imshow("x", cv.cvtColor(x, cv.COLOR_RGB2BGR).astype(np.uint8))
        cv.waitKey(1)

def runTestModeFullAuto(modelArch,categorical):
    img_shape = (480, 480, 3)
    num_cat = 6
    batchSize = 4
    max_epoch = 150

    calc_downSample = 8
    plottingUpSample = 1 / 4

    dataset = Generators.SyntheticDataSet(None, None, None, img_shape, num_cat)

    pwd = os.path.dirname(os.path.abspath(sys.argv[0]))

    modelSaveFolderName = getSyntheticTrainedModelNames(modelArch, categorical)
    outPutFolderPath = os.path.join(pwd, "thesisModels", modelSaveFolderName)
    modelLoc = os.path.join(outPutFolderPath, 'model')
    model = k.models.load_model(modelLoc)

    s = UnityServer.UnityServer()
    servThread = threading.Thread(target=s.update)
    servThread.start()

    print("wait to recive first image")
    while s.state[1] is None:
        time.sleep(.1)
    print("got first image")

    def costPredictor(x):
        x = cv.resize(x, img_shape[:2])
        x = np.expand_dims(x, axis=0)
        y = model.predict(x)

        # x = np.squeeze(x)
        y = np.squeeze(y)

        if categorical:
            y = dataset.costMapFromEncoded(np.argmax(y, axis=2).astype(np.float32))

        return dataset.scaleNormedCostMap(y)

    streamStitcher = ImageStreamStitcher(isaffine=True)
    p, cm = streamStitcher.consumeStream(s, costPredictor,max(dataset.weights))

    #dont need the seg model anymore
    del model

    w = WaveFront(p, cm, dataset, calc_downSample)
    hm = w.getPCMHeatMapPlot(plottingUpSample)
    w.manualSetEndPoints(plottingUpSample,
                         imgPlot=w.getImgAtHeatPlotSize(plottingUpSample),
                         heatMapPlot=hm,
                         plotName="wave")
    w.generateWaveCostMap(plottingUpSample=plottingUpSample,
                          plotName="WaveFrontHeatMap",
                          plot=True)

    while not s.done:
        img = s.state[1]
        # cv.imshow('UAV OBS',img)
        localCostMap = w.getLocalCostMapFromTemplate(img, plottingUpSample)

        # simply assume the ugv is in the center of the image alternativly this could be provided by unity or some other module
        sx = img.shape[WaveFront.xidx] / 2 // w.calc_downSample
        sy = img.shape[WaveFront.yidx] / 2 // w.calc_downSample

        plot, localPath, done = w.getLocalPathFromLocalCostMap(sx, sy, img, localCostMap)

        if done:
            print("Goal reched setting done")
            s.action = 'done'

            #probably a better way to do this but waiting to ensure python sends done before exit
            time.sleep(1)
            s.done = True


        force = .4
        delta = np.array(localPath[0]) - np.array([sx, sy])
        print(delta)
        # open cv to unity cordinate system adding pi over 2
        heading = np.rad2deg(np.arctan2(delta[1], delta[0]) + np.pi / 2)

        print(heading)
        s.action = (heading, force)

if __name__ == "__main__":
    import os
    import sys

    import Generators
    import Models
    import UnityServer
    import scratch

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

    modelArch = 'deeplab'
    categorical = True

    modelSaveFolderName = getSyntheticTrainedModelNames(modelArch, categorical)

    ############################################################################
    ##############################RUN IN TRAINING MODE##########################

    dataFolder = os.path.join("E:", "UnitySegOutPut", "generatedData1")
    # # dataFolder = os.path.join("/home", "samiw", "thesis", "data", "UnitySegOutPut","generatedDataCOPY")
    #
    # #######Run Training###########
    # runTrainMode(modelArch, categorical, modelSaveFolderName, dataFolder)
    #
    # ###########view outputs of trained model on flood net#########
    # visModelOutForFloodNet(modelSaveFolderName,categorical)
    # visModelOutForSynthData(modelSaveFolderName, dataFolder, categorical)

    ##spit out eval metrics
    # getEvalMetricsForModel(modelSaveFolderName,dataFolder,categorical)

    ##########################END TRAIN MODE######################################
    ##############################################################################

    ##############################################################################
    #########################RUN TEST MODE########################################


    runTestModeFullAuto(modelArch,categorical)

    # folderPath = os.path.join("E:", "UnitySegOutPut", "testSenarioJacksonville")
    # runClassificationOnFolder(categorical, modelSaveFolderName,folderPath)

    # imgs2ClassFolder = os.path.join("E:","UnitySegOutPut", "testSenarioJacksonville")
    # generateateNavResultForAllModelsAtLoc(imgs2ClassFolder, "AllPathsJacksonville.png")



#############################################

    # fdir = os.path.join("C:\\", "Users", "samiw", "OneDrive", "Desktop", "Desktop", "VT", "Research", "imageStitch",
    #                     "testOutPuts", "full_res")
    #
    # xpath = os.path.join(fdir, "X_pano.png")
    # pixelCostpath = os.path.join(fdir, "cm_pano.npy")
    # calc_downSample = 8
    # plottingUpSample = 1 / 2
    #
    # img_shape = (480, 480, 3)
    # num_cat = 6
    # d = Generators.SyntheticDataSet(None, None, None, img_shape, num_cat)
    #
    # with open(pixelCostpath, 'rb') as f:
    #     pcm = np.load(f)
    #
    # ####DO NOT FEED TO KERAS LIKE THIS
    # x = cv.imread(xpath)
    #
    # w = WaveFront(x, pcm, d, calc_downSample)
    # w.readWaveCostMapFromFile('norfolkTestCostMap.npy')
    #
    # s = UnityServer.UnityServer()
    # servThread = threading.Thread(target=s.update)
    # servThread.start()
    #
    # print("wait to recive first image")
    # while s.state[1] is None:
    #     time.sleep(.1)
    # print("got first image")
    #
    # done =False
    #
    # while not done:
    #     img = s.state[1]
    #     #cv.imshow('UAV OBS',img)
    #     localCostMap = w.getLocalCostMapFromTemplate(img, plottingUpSample)
    #
    #     #simply assume the ugv is in the center of the image alternativly this could be provided by unity or some other module
    #     sx = img.shape[WaveFront.xidx] / 2 // w.calc_downSample
    #     sy = img.shape[WaveFront.yidx] / 2 // w.calc_downSample
    #
    #     plot, localPath, done = w.getLocalPathFromLocalCostMap(sx, sy, img, localCostMap)
    #
    #     force = .4
    #     delta = np.array(localPath[0]) - np.array([sx,sy])
    #     print(delta)
    #     #open cv to unity cordinate system adding pi over 2
    #     heading = np.rad2deg(np.arctan2(delta[1],delta[0]) + np.pi/2)
    #
    #
    #     print(heading)
    #     s.action = (heading,force)
    ###
    # nano thesis/code/DeepLearnFinalProject/Driver.py
    # "C:\Program Files\PuTTY\psftp.exe" samiw@ada.hume.vt.edu - P 2200 - i C:\Users\samiw\OneDrive\Desktop\Desktop\VT\Research\summer2020\raytheonSSHKeys\rayPrivateKeyForUseWithPutty.ppk

