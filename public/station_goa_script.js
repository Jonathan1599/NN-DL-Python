let x = [];
let y = [];
let z = [];

let labels = [0];
console.log("hi");

var ctx = document.getElementById("myChart").getContext("2d");

var chart1 = new Chart(ctx, {
  // The type of chart we want to create
  type: "line",

  // The data for our dataset
  data: {
    labels: labels,
    datasets: [
      {
        label: "X",
        //backgroundColor: 'rgb(255, 99, 132)',
        borderColor: "rgb(255, 99, 132)",
        data: x,
      },
      {
        label: "Y",
        //backgroundColor: 'rgb(255, 99, 132)',
        borderColor: "rgb(0, 99, 132)",
        data: y,   
      },
    ],
  },

  options: {
    scales: {
      yAxes: [
        {
          ticks: {
            beginAtZero: true,
          },
        },
      ],
    },
  },
});

let socket = io.connect("http://localhost:9000");

socket.on("goa", (data) => {
  console.log(data);
  x.push(data.x);
  y.push(data.y);
  z.push(data.z);
  labels.push(labels[labels.length - 1] + 1);
  chart1.update();
});

// const ctx = document.getElementById('myChart');
// const myChart = new Chart(ctx, {
//     type: 'line',
//     data: {
//         labels: ['Red', 'Blue',  'Green'],
//         datasets: [{
//             label: '# of Votes',
//             data: [12, 19, 3, 5, 2, 3],
//             backgroundColor: [
//                 'rgba(255, 99, 132, 0.2)',
//                 'rgba(54, 162, 235, 0.2)',
//                 'rgba(255, 206, 86, 0.2)',

//             ],
//             borderColor: [
//                 'rgba(255, 99, 132, 1)',
//                 'rgba(54, 162, 235, 1)',
//                 'rgba(255, 206, 86, 1)',
//             ],
//             borderWidth: 1
//         }]
//     },
//     options: {
//         scales: {
//             y: {
//                 beginAtZero: true
//             }
//         }
//     }
// })
