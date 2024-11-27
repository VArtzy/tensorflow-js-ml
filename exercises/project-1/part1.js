import * as tf from "@tensorflow/tfjs"
import * as cocoSsd from "@tensorflow-models/coco-ssd"
import { handleFilePicker, showResult } from "./utils"

const predict = async (imgElement) => {
    let model = await cocoSsd.load()
    const predictions = await model.detect(imgElement)

    showResult(predictions)
}

handleFilePicker(predict)
