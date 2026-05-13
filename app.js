// =======================================
// CONFIG
// =======================================

const CONFIG = {

    modelPath: "./best.onnx",

    labels: [
        "lighting",
        "micro_usb",
        "type-c"
    ],

    threshold: 0.45
};

// =======================================
// HTML ELEMENT
// =======================================

const video = document.getElementById("webcam");

const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");

const processor = document.getElementById("processor");
const pctx = processor.getContext("2d");

const statusDiv = document.getElementById("status");

const initBtn = document.getElementById("btn-init");

// =======================================
// GLOBAL
// =======================================

let session;

// =======================================
// LOAD MODEL
// =======================================

async function loadModel() {

    try {

        statusDiv.innerHTML = "Loading model AI...";

        session = await ort.InferenceSession.create(
            CONFIG.modelPath
        );

        console.log("MODEL LOADED");

        statusDiv.innerHTML =
            "Model AI berhasil dimuat";

    } catch (err) {

        console.error(err);

        statusDiv.innerHTML =
            "Gagal load model: " + err;
    }
}

// =======================================
// START CAMERA
// =======================================

async function startCamera() {

    try {

        const stream =
            await navigator.mediaDevices.getUserMedia({
                video: true
            });

        video.srcObject = stream;

        statusDiv.innerHTML =
            "Kamera aktif";

        detectLoop();

    } catch (err) {

        console.error(err);

        statusDiv.innerHTML =
            "Kamera gagal diakses";
    }
}

// =======================================
// PREPROCESS IMAGE
// =======================================

function preprocessFrame() {

    pctx.drawImage(video, 0, 0, 640, 480);

    const imageData =
        pctx.getImageData(0, 0, 640, 480);

    const { data } = imageData;

    const input =
        new Float32Array(640 * 640 * 3);

    let j = 0;

    for (let i = 0; i < data.length; i += 4) {

        input[j++] = data[i] / 255.0;
        input[j++] = data[i + 1] / 255.0;
        input[j++] = data[i + 2] / 255.0;
    }

    return input;
}

// =======================================
// DETECTION LOOP
// =======================================

async function detectLoop() {

    if (!session) return;

    try {

        const inputData =
            preprocessFrame();

        const tensor =
            new ort.Tensor(
                "float32",
                inputData,
                [1, 3, 640, 640]
            );

        // Nama input model
        // Kadang:
        // images
        // input
        // input0

        const feeds = {
            images: tensor
        };

        const results =
            await session.run(feeds);

        console.log(results);

        drawFakeDetection();

    } catch (err) {

        console.error(err);

        statusDiv.innerHTML =
            "Inference error";
    }

    requestAnimationFrame(detectLoop);
}

// =======================================
// DRAW RESULT
// =======================================

function drawFakeDetection() {

    ctx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    ctx.strokeStyle = "lime";
    ctx.lineWidth = 3;

    ctx.strokeRect(
        200,
        100,
        220,
        200
    );

    ctx.font = "20px Arial";

    ctx.fillStyle = "lime";

    ctx.fillText(
        "Kabel Terdeteksi",
        200,
        90
    );
}

// =======================================
// BUTTON EVENT
// =======================================

initBtn.addEventListener(
    "click",
    async () => {

        await loadModel();

        await startCamera();
    }
);
