const canvasElement = document.getElementById("draw-area")
const canvas = new HandwritingCanvas(canvasElement)
const clearButtonElement = document.getElementById("clear-button")
clearButtonElement.addEventListener("click", () => {
  canvas.clear()
})

const predictButtonElement = document.getElementById("predict-button")

async function preprocess(blob) {
  // 画像を28 x 28に変換
  const canvas = document.createElement("canvas")
  const ctx = canvas.getContext("2d")
  canvas.height = 28
  canvas.width = 28

  const bitmap = await createImageBitmap(blob, {
    resizeHeight: 28,
    resizeWidth: 28,
  })
  ctx.drawImage(bitmap, 0, 0)
  const imageData = ctx.getImageData(0, 0, 28, 28)
  // RGBAのA(alpha)のみを取り出す
  const alphas = []
  for (let i = 0; i < imageData.data.length; i++) {
    if (i % 4 == 3) {
      const alpha = imageData.data[i]
      alphas.push(alpha)
    }
  }
  return alphas
}

async function predict(input) {
  const session = await ort.InferenceSession.create("/static/model.onnx")
  const feeds = {
    float_input: new ort.Tensor("float32", input, [1, 28 * 28]),
  }
  const results = await session.run(feeds)
  return results.probabilities.data
}

predictButtonElement.addEventListener("click", async () => {
  if (canvas.isEmpty) {
    return
  }
  // 推論実行
  const blob = await canvas.toBlob("image/png")
  const input = await preprocess(blob)
  const probabilities = await predict(input)

  // 推論結果の画像を表示
  const imageUrl = URL.createObjectURL(blob)
  const imgElement = document.createElement("img")
  imgElement.src = imageUrl
  const resultImageElement = document.getElementById("result-image")
  if (resultImageElement.firstChild) {
    resultImageElement.removeChild(resultImageElement.firstChild)
  }
  resultImageElement.append(imgElement)
  canvas.clear()

  // 推論結果をtbodyに表示
  const tableBodyElement = document.getElementById("result-table-body")
  while (tableBodyElement.firstChild) {
    tableBodyElement.removeChild(tableBodyElement.firstChild)
  }
  for (let i = 0; i < probabilities.length; i++) {
    const tr = document.createElement("tr")
    // 数字
    const tdNumber = document.createElement("td")
    tdNumber.textContent = i
    tr.appendChild(tdNumber)
    // 確率
    const tdProbability = document.createElement("td")
    tdProbability.textContent = (probabilities[i] * 100).toFixed(1)
    tr.appendChild(tdProbability)
    tableBodyElement.appendChild(tr)
  }
})
