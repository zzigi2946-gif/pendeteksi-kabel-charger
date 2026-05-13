// ======================================================================
// KONFIGURASI
// ======================================================================

const CONFIG = {

    // Lokasi model ONNX
    modelPath: './best.onnx',

    // Label kelas
    // Sesuaikan dengan urutan label saat training
    labels: [
        "Kelas_Satu",
        "Kelas_Dua"
    ],

    // Confidence threshold
    threshold: 0.45,

    // IoU threshold untuk NMS
    iouThreshold: 0.4
};

// ======================================================================
// ELEMENT HTML
// ======================================================================

const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctxOverlay = overlay.getContext('2d');

const processor = document.getElementById('processor');
const ctxProcessor = processor.getContext('2d', {
    willReadFrequently: true
});

const status = document.getElementById('status');
const initBtn = document.getElementById('btn-init');

// ======================================================================
// VARIABEL
// ======================================================================

let session = null;

const TARGET_SIZE = 640;

// ======================================================================
// LOAD MODEL
// ======================================================================

initBtn.addEventListener('click', async () => {

    initBtn.disabled = true;
    initBtn.innerText = "MEMUAT MODEL AI...";

    try {

        status.innerText = "MEMUAT best.onnx ...";

        // Path WASM ONNX Runtime
        ort.env.wasm.wasmPaths =
            'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

        // Load model
        session = await ort.InferenceSession.create(
            CONFIG.modelPath,
            {
                executionProviders: ['webgl', 'wasm']
            }
        );

        status.innerText = "MODEL BERHASIL DIMUAT";

        // Start kamera
        startCamera();

    } catch (error) {

        console.error(error);

        status.innerText =
            "GAGAL MEMUAT MODEL best.onnx";

        initBtn.disabled = false;
        initBtn.innerText = "AKTIFKAN KAMERA";
    }
});

// ======================================================================
// START CAMERA
// ======================================================================

async function startCamera() {

    try {

        const stream =
            await navigator.mediaDevices.getUserMedia({
                video: {
                    width: 640,
                    height: 480
                },
                audio: false
            });

        video.srcObject = stream;

        video.onloadedmetadata = () => {

            video.play();

            status.innerText =
                "KAMERA AKTIF - DETEKSI BERJALAN";

            initBtn.style.display = "none";

            requestAnimationFrame(processFrame);
        };

    } catch (error) {

        console.error(error);

        status.innerText =
            "GAGAL MENGAKSES KAMERA";
    }
}

// ======================================================================
// LOOP DETEKSI
// ======================================================================

async function processFrame() {

    if (!session) return;

    // ==========================================================
    // Ambil frame kamera
    // ==========================================================

    ctxProcessor.drawImage(
        video,
        0,
        0,
        TARGET_SIZE,
        TARGET_SIZE
    );

    const imageData =
        ctxProcessor.getImageData(
            0,
            0,
            TARGET_SIZE,
            TARGET_SIZE
        ).data;

    // ==========================================================
    // Convert RGB ke Float32
    // ==========================================================

    const float32Data =
        new Float32Array(
            3 * TARGET_SIZE * TARGET_SIZE
        );

    for (let i = 0; i < TARGET_SIZE * TARGET_SIZE; i++) {

        float32Data[i] =
            imageData[i * 4] / 255.0;

        float32Data[
            i + TARGET_SIZE * TARGET_SIZE
        ] =
            imageData[i * 4 + 1] / 255.0;

        float32Data[
            i + 2 * TARGET_SIZE * TARGET_SIZE
        ] =
            imageData[i * 4 + 2] / 255.0;
    }

    // ==========================================================
    // Tensor Input
    // ==========================================================

    const inputTensor = new ort.Tensor(
        'float32',
        float32Data,
        [1, 3, TARGET_SIZE, TARGET_SIZE]
    );

    // ==========================================================
    // Inference
    // ==========================================================

    const results = await session.run({
        [session.inputNames[0]]: inputTensor
    });

    const output =
        results[session.outputNames[0]].data;

    // ==========================================================
    // Parsing Output YOLO
    // ==========================================================

    const numClasses =
        CONFIG.labels.length;

    const elements = 8400;

    let rawBoxes = [];

    for (let i = 0; i < elements; i++) {

        let maxScore = 0;
        let classId = -1;

        // Cari class terbaik
        for (let c = 0; c < numClasses; c++) {

            const score =
                output[
                    i + (4 + c) * elements
                ];

            if (score > maxScore) {

                maxScore = score;
                classId = c;
            }
        }

        // Filter threshold
        if (maxScore > CONFIG.threshold) {

            let x = output[i];
            let y = output[i + elements];
            let w = output[i + 2 * elements];
            let h = output[i + 3 * elements];

            // Jika output normalize
            if (w <= 1.5) {

                x *= TARGET_SIZE;
                y *= TARGET_SIZE;
                w *= TARGET_SIZE;
                h *= TARGET_SIZE;
            }

            rawBoxes.push({

                x: x - w / 2,
                y: y - h / 2,

                w: w,
                h: h,

                score: maxScore,

                classId: classId
            });
        }
    }

    // ==========================================================
    // NMS
    // ==========================================================

    const finalBoxes =
        nonMaxSuppression(
            rawBoxes,
            CONFIG.iouThreshold
        );

    // ==========================================================
    // Draw
    // ==========================================================

    drawBoxes(finalBoxes);

    requestAnimationFrame(processFrame);
}

// ======================================================================
// IOU
// ======================================================================

function calculateIoU(box1, box2) {

    const xA =
        Math.max(box1.x, box2.x);

    const yA =
        Math.max(box1.y, box2.y);

    const xB =
        Math.min(
            box1.x + box1.w,
            box2.x + box2.w
        );

    const yB =
        Math.min(
            box1.y + box1.h,
            box2.y + box2.h
        );

    const intersectionArea =
        Math.max(0, xB - xA) *
        Math.max(0, yB - yA);

    return (
        intersectionArea /
        (
            (box1.w * box1.h) +
            (box2.w * box2.h) -
            intersectionArea
        )
    );
}

// ======================================================================
// NON MAX SUPPRESSION
// ======================================================================

function nonMaxSuppression(boxes, iouThreshold) {

    boxes.sort((a, b) =>
        b.score - a.score
    );

    const result = [];

    while (boxes.length > 0) {

        const current = boxes.shift();

        result.push(current);

        boxes = boxes.filter(box =>
            calculateIoU(current, box)
            < iouThreshold
        );
    }

    return result;
}

// ======================================================================
// DRAW BOX
// ======================================================================

function drawBoxes(boxes) {

    ctxOverlay.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    boxes.forEach(box => {

        const scaleX =
            overlay.width / TARGET_SIZE;

        const scaleY =
            overlay.height / TARGET_SIZE;

        const x = box.x * scaleX;
        const y = box.y * scaleY;

        const w = box.w * scaleX;
        const h = box.h * scaleY;

        // Kotak
        ctxOverlay.strokeStyle = "#34C759";
        ctxOverlay.lineWidth = 3;

        ctxOverlay.strokeRect(
            x,
            y,
            w,
            h
        );

        // Background label
        ctxOverlay.fillStyle = "#34C759";

        ctxOverlay.fillRect(
            x,
            y - 25,
            170,
            25
        );

        // Teks label
        ctxOverlay.fillStyle = "black";

        ctxOverlay.font =
            "bold 16px Arial";

        ctxOverlay.fillText(
            `${CONFIG.labels[box.classId]} ${(box.score * 100).toFixed(1)}%`,
            x + 5,
            y - 7
        );
    });
}
