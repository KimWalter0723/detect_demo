document.getElementById("startCameraButton").addEventListener("click", startCamera);
document.getElementById("startDetectionButton").addEventListener("click", startDetection);
document.getElementById("stopDetectionButton").addEventListener("click", stopDetection);

let session = null;  
let isDetecting = false;  

async function startCamera() {
    const video = document.getElementById("video");
    
    // 获取摄像头权限
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            document.getElementById("startDetectionButton").disabled = false; 
        })
        .catch(err => console.error("无法访问摄像头", err));

    // 预加载 YOLO ONNX 模型（从根目录加载）
    session = await ort.InferenceSession.create('/best.onnx', {
        executionProviders: ['webgl']
    });

    console.log("YOLO 模型加载完成");
}

async function startDetection() {
    if (!session) {
        alert("模型尚未加载，请稍候...");
        return;
    }

    isDetecting = true;
    document.getElementById("startDetectionButton").disabled = true;
    document.getElementById("stopDetectionButton").disabled = false;
    detectObjects();
}

function stopDetection() {
    isDetecting = false;
    document.getElementById("startDetectionButton").disabled = false;
    document.getElementById("stopDetectionButton").disabled = true;
}

async function detectObjects() {
    if (!isDetecting) return;

    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const inputTensor = new ort.Tensor('float32', imageData.data, [1, 3, canvas.height, canvas.width]);

    const results = await session.run({ 'input': inputTensor });

    displayResults(results);
    requestAnimationFrame(detectObjects);
}

// 显示识别结果
function displayResults(results) {
    const resultBox = document.getElementById("resultsList");
    resultBox.innerHTML = "";  

    results.output.forEach(item => {
        const listItem = document.createElement("li");
        listItem.textContent = `类别: ${item.class}, 置信度: ${(item.confidence * 100).toFixed(2)}%`;
        resultBox.appendChild(listItem);
    });
}
