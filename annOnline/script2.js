let canvas1 = document.getElementById('canvas1');
let ctx1 = canvas1.getContext('2d');
let canvas2 = document.getElementById('canvas2');
let ctx2 = canvas2.getContext('2d');
let points = [];
const learningRate = 0.00001;

function generatePoints(numPoints, width, height) {
    let pts = [];
    for (let i = 0; i < numPoints; i++) {
        let x = Math.floor(Math.random() * width);
        let y = Math.floor(Math.random() * height);
        pts.push({x: x, y: y});
    }
    return pts;
}

function clearCanvas1() {
    ctx1.clearRect(0, 0, canvas1.width, canvas1.height);
}

function clearCanvas2() {
    ctx2.clearRect(0, 0, canvas1.width, canvas1.height);
}

function drawPoints() {
    clearCanvas1();
    for (let point of points) {
        ctx2.fillStyle = 'black';
        ctx1.beginPath();
        ctx1.arc(point.x, canvas1.height - point.y, 5, 0, Math.PI * 2);
        ctx1.fill();
    }
}

function drawPoints2() {
    clearCanvas2();
    for (let point of points) {
        ctx2.fillStyle = 'black';
        ctx2.beginPath();
        ctx2.arc(point.x, canvas1.height - point.y, 5, 0, Math.PI * 2);
        ctx2.fill();
    }
}

function drawLine() {
    let p1 = {x: 0, y: 0};
    let p2 = {x: 0, y: 0};
    let slope = 0;
    let intercept = 0;
    canvas1.addEventListener('mousedown', function(event) {
        let rect = canvas1.getBoundingClientRect();
        p1.x = event.clientX - rect.left;
        p1.y = canvas1.height - (event.clientY - rect.top);
        canvas1.addEventListener('mouseup', function(event) {
            p2.x = event.clientX - rect.left;
            p2.y = canvas1.height - (event.clientY - rect.top);
            slope = (p2.y - p1.y) / (p2.x - p1.x);
            intercept = p1.y - slope * p1.x;
            // document.getElementById('slope').textContent = slope.toFixed(2);
            // document.getElementById('intercept').textContent = intercept.toFixed(2);
            document.getElementById('slope').textContent = slope;
            document.getElementById('intercept').textContent = intercept;
            ctx1.beginPath();
            ctx1.moveTo(p1.x, canvas1.height - p1.y);
            ctx1.lineTo(p2.x, canvas1.height - p2.y);
            ctx1.stroke();
            for (let i = 0; i < points.length; i++) {
				let point = points[i];
                if (point.y > slope * point.x + intercept + 5) {
                    ctx1.fillStyle = 'red';
                } else {
                    ctx1.fillStyle = 'green';
                }
                ctx1.beginPath();
                ctx1.arc(point.x, canvas1.height - point.y, 5, 0, Math.PI * 2);
                ctx1.fill();
            }
        });
    });
}

// Line Function to be used based on the user's input line. This function generates the target training data based on which the perceptron is trained
function f(x, slope, intercept) {
    var y = parseFloat(x * slope) + parseFloat(intercept);
    return y
}

function trainPerceptron(slope, intercept) {
	// Compute Target Data (desired answers)
	const desired = [];
	for (let i = 0; i < points.length; i++) {
		let point = points[i];
		desired[i] = 0.0;
		if (point.y > f(point.x, slope, intercept)) {desired[i] = 1.0}
		}
	// Create a Perceptron
	const ptron = new Perceptron(2, learningRate);
	// Train the Perceptron
	for (let j = 0; j <= 10000; j++) {
		for (let i = 0; i < points.length; i++) {
			let point = points[i];
			ptron.train([point.x, point.y], desired[i]);
		}
	}
	return ptron
}
	
	// Display the Result
function applyPerceptron(points,ptron) {
	for (let i = 0; i < points.length; i++) {
		let point = points[i];
		const x = point.x;
		const y = point.y;
		let guess = ptron.activate([x, y, ptron.bias]);
		ctx2.beginPath();
        	ctx2.arc(point.x, canvas2.height - point.y, 5, 0, Math.PI * 2);
		let color = 'red';
		if (guess == 0) color = 'green';
		ctx2.fillStyle = color;
        	ctx2.fill();
        	// document.getElementById("demo").innerHTML = guess;
	}
}


document.getElementById('generate-button').addEventListener('click', function() {
    let numPoints = parseInt(document.getElementById('num-points-input').value);
    let width = canvas1.width;
    let height = canvas1.height;
    points = generatePoints(numPoints, width, height);
    drawPoints();
});

document.getElementById('generate-button-2').addEventListener('click', function() {
    let numPoints = parseInt(document.getElementById('num-points-input').value);
    let width = canvas2.width;
    let height = canvas2.height;
    points = generatePoints(numPoints, width, height);
    drawPoints2();
});

document.getElementById('draw-button').addEventListener('click', function() {
    /* drawLine(); */
	var line1 = drawLine(); // call drawLine function and store the result in line1 variable
});

document.getElementById('clear-button').addEventListener('click', function() {
    clearCanvas1();
    document.getElementById('slope').textContent = '';
    document.getElementById('intercept').textContent = '';
});

document.getElementById('train-button').addEventListener('click', function() {
    var slope = document.getElementById('slope').textContent;
    var intercept = document.getElementById('intercept').textContent;
    ptron=trainPerceptron(slope, intercept);
});

document.getElementById('apply-button').addEventListener('click', function() {
    let numPoints = parseInt(document.getElementById('num-points-input').value);
    let width = canvas2.width;
    let height = canvas2.height;
    applyPerceptron(points,ptron)
});




