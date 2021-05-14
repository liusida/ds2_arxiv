document.addEventListener("DOMContentLoaded", function(event) {
    "use strict";

    d3.selectAll(".paper")
        .style("top",
               function() {
                   let _top =
                       parseInt(parseFloat(d3.select(this).attr('index') * 12));
                   return _top + "px";
               })
        .on("mousemove touchmove",
            function(d) {
                let _arxiv_id = d3.select(this).attr("arxiv_id");
                let _tooltip = d3.select(this).attr("tooltip");
                let _top =
                    parseInt(parseFloat(d3.select(this).attr('index') * 12));

                d3.select("#arxiv_tooltip")
                    .style("font-size", parseInt(20 / detectZoom.zoom()) + "px")
                    .style("width", parseInt(500 / detectZoom.zoom()) + "px")
                    .style("top", (_top + 20) + "px")
                    .style("left", d.pageX + "px")
                    .html(_tooltip);

                d3.select("#arrow").style("top", _top + "px");
            })
        .on("mouseout",
            function() { d3.select("#arxiv_tooltip").style("top", "-1000px") })
        .on("click", function(d) {
            let _arxiv_id = d3.select(this).attr("arxiv_id");
            window.open("https://arxiv.org/abs/" + _arxiv_id, "_blank");
        });
});