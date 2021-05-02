<!DOCTYPE html>
<html>
<?php
$bgscale = 8.485;
?>

<head>
    <meta charset='utf-8'>
    <title>Deep Learning</title>
    <script src="https://d3js.org/d3.v6.min.js"></script>
    <script src="detect-zoom.js"></script>
    <style>
        #main {
            /* border: 1px solid red; */
            position: relative;
            overflow: hidden;
            height: 53300px;
        }

        #bg {
            position: absolute;
            top: 6px;
            left: 5px;
            height: <?php echo 4422 * $bgscale ?>px;
            width: <?php echo 4422 * $bgscale ?>px;
            background-image: url('bg.png');
            background-size: <?php echo 4422 * $bgscale ?>px;
            transform-origin: top left;
            transform: rotate(45deg);
            z-index: -1;
        }

        #list {
            padding-top: 20px;
            padding-left: 0px;
            margin-left: 500px;
            position: relative;
        }

        #list div {
            position: absolute;
            top: 0px;
            left: 0px;
            margin-left: 0px;
            padding-left: 0px;
            line-height: 13px;
            font-size: 8px;
            white-space: nowrap;
            background-color: white;
            cursor: pointer;
        }

        #arxiv_tooltip {
            border: 1px solid black;
            background-color: white;
            position: absolute;
            top: -1000px;
            left: -1000px;
            width: 500px;
            /* height: 200px; */
            padding: 20px;
        }

        #arrow {
            color: red;
            font-size: 20px;
            position: absolute;
            top: -1000px;
            left: 0px;
        }


        #test {
            display: none;
            margin: 100px;
        }

        #test #outer_square {
            height: 100px;
            width: 100px;
            background-color: green;
        }

        #test #inner_square {
            height: 100px;
            width: 100px;
            background-color: red;
            transform-origin: top left;
            transform: rotate(45deg);
        }
    </style>

    <script>
        document.addEventListener("DOMContentLoaded", function(event) {
            // return;
            d3.selectAll(".paper").style("top", function() {
                    let _top = parseInt(parseFloat(d3.select(this).attr('index') * 12));
                    return _top + "px";
                })
                .on("mousemove", function(d) {
                    let _arxiv_id = d3.select(this).attr("arxiv_id");
                    let _tooltip = d3.select(this).attr("tooltip");
                    let _top = parseInt(parseFloat(d3.select(this).attr('index') * 12));

                    d3.select("#arxiv_tooltip")
                        .style("font-size", parseInt(20 / detectZoom.zoom())+"px")
                        .style("width", parseInt(500 / detectZoom.zoom())+"px")
                        .style("top", d.pageY + "px")
                        .style("left", d.pageX + "px")
                        .html(_tooltip);
                    
                    d3.select("#arrow")
                        .style("top", _top + "px");

                })
                .on("mouseout", function(){
                    d3.select("#arxiv_tooltip")
                        .style("top", "-1000px")
                })
                .on("click", function(d) {
                    let _arxiv_id = d3.select(this).attr("arxiv_id");
                    window.open("https://arxiv.org/abs/"+_arxiv_id, "_blank");
                });


        });
    </script>

</head>

<body>
    <div id="test">
        <div id="outer_square">
            <div id="inner_square"></div>
        </div>
    </div>
    <div id="main">
        <div id="bg">
        </div>
        <div id="list">
            <?php include 'the_page.html'; ?>
        </div>
        <div id="arxiv_tooltip"></div>
        <div id="arrow">➞</div>
    </div>
</body>

</html>