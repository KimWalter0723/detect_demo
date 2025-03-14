document.getElementById("startButton").addEventListener("click", startCamera);

async function startCamera() {
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    // 访问摄像头
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => { video.srcObject = stream; })
        .catch(err => console.error("无法访问摄像头", err));

    // 加载 YOLO ONNX 模型
    const session = await ort.InferenceSession.create('best.onnx', {
        executionProviders: ['webgl']
    });

    async function detectObjects() {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // 获取图像数据
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const inputTensor = new ort.Tensor('float32', imageData.data, [1, 3, canvas.height, canvas.width]);

        // 运行 YOLO 推理
        const results = await session.run({ 'input': inputTensor });

        displayResults(results);
        requestAnimationFrame(detectObjects);
    }

    detectObjects();
}

// 显示识别结果
function displayResults(results) {
    const resultBox = document.getElementById("resultsList");
    resultBox.innerHTML = "";  // 清空旧结果

    results.output.forEach(item => {
        const listItem = document.createElement("li");
        listItem.textContent = `类别: ${item.class}, 置信度: ${(item.confidence * 100).toFixed(2)}%`;
        resultBox.appendChild(listItem);
    });
}
