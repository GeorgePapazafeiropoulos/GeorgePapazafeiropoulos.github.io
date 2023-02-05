// Adding function
function add()
{
  var numOne, numTwo, numRes;
  numOne = parseInt(document.getElementById("first").value);
  numTwo = parseInt(document.getElementById("second").value);
  numRes = numOne + numTwo;
  document.getElementById("answer").value = numRes;
}

// Subtracting function
function subtract()
{
  var numOne, numTwo, numRes;
  numOne = parseInt(document.getElementById("first").value);
  numTwo = parseInt(document.getElementById("second").value);
  numRes = numOne - numTwo;
  document.getElementById("answer").value = numRes;
}

// Multiplication function
function multiply()
{
  var numOne, numTwo, numRes;
  numOne = parseInt(document.getElementById("first").value);
  numTwo = parseInt(document.getElementById("second").value);
  numRes = numOne * numTwo;
  document.getElementById("answer").value = numRes;
}

// Division function
function divide()
{
  var numOne, numTwo, numRes;
  numOne = parseInt(document.getElementById("first").value);
  numTwo = parseInt(document.getElementById("second").value);
  numRes = numOne / numTwo;
  document.getElementById("answer").value = numRes;
}
