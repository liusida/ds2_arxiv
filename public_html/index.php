<!DOCTYPE html>
<html>

<head>
    <meta charset='utf-8'>
    <title>Deep Learning</title>
    <script src="https://d3js.org/d3.v6.min.js"></script>
    <script src="js/detect-zoom.js"></script>
    <script src="js/reordering.js"></script>
    <script src='js/d3.layout.cloud.js'></script>
    <script src="js/world_cloud.js"></script>
    <link rel="stylesheet" href="css/main.css">
    <style>
        <?php
        $bgscale = 8.4855;
        ?>
        #main {
            /* border: 1px solid red; */
            position: relative;
            overflow: hidden;
            height: <?php echo 3016 * $bgscale + 12000 ?>px;
        }
        #bg {
            position: absolute;
            top: 6px;
            left: 5px;
            height: <?php echo 3016 * $bgscale ?>px;
            width: <?php echo 3016 * $bgscale ?>px;
            background-image: url('bg.png');
            background-size: <?php echo 3016 * $bgscale ?>px <?php echo 3016 * $bgscale ?>px;
            transform-origin: top left;
            transform: rotate(45deg);
            z-index: -1;
        }
    </style>
</head>

<body>
    <div id="intro">
        <p class="title">Exploring Highly Cited arXiv Deep Learning Papers</p>
        <p>This list shows 4,422 highly cited arXiv papers in two subcategories: cs.LG and cs.AI retrived in April, 2021.</p>
        <p>Refer to <a href="paper.pdf" target="_blank">this paper</a> for details.
        <p>The order of the list is arranged so that papers with stronger relationships are placed closer to each other.</p>
        <p>Here is a preview of the entire matrix. The most famous authors are usually in <a href="#paper_2080">the middle of the list</a>.</p>
        <div class="img"><a href="bg.png" target="_blank"><img src="bg.png" width="400" height="400"></a></div>
        <p>For more information, please refer to: <a href="https://github.com/liusida/arxiv_4422">https://github.com/liusida/arxiv_4422</a>
        <p>Please zoom out to see the whole picture and zoom in to see the individual papers.</p>
        <p>The detailed list starts here:</p>
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
    <div id="word-cloud-word-list">
        <?php include("word_cloud.html") ?>
    </div>
    <div id="explain-word-cloud">
        <p>You can also click the words bellow to highlight the papers which contains the word.</p>
        <p>This is helpful when everything is small.</p>
        <p>You can choose multiple words with logic: <br>
            <input type="radio" name="word-cloud-logic" value="or" checked> OR<br>
            <input type="radio" name="word-cloud-logic" value="and"> AND<br>
        </p>
    </div>
    <div id="word-cloud"></div>
</body>

</html>