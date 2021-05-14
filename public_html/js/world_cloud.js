document.addEventListener("DOMContentLoaded", function(event) {
    "use strict";

    Array.prototype.remove = function() {
        var what, a = arguments, L = a.length, ax;
        while (L && this.length) {
            what = a[--L];
            while ((ax = this.indexOf(what)) !== -1) {
                this.splice(ax, 1);
            }
        }
        return this;
    };

    var highlighted_words = [];
    var words = d3.select("#word-cloud-word-list")
                    .text()
                    .split("\n")
                    .map(function(s) { return s.trim(); })
                    .filter(function(s) { return s.indexOf(":::") > 0; });

    var layout = d3.layout.cloud()
                     .size([ 2000, 1000 ])
                     .words(words.map(function(d) {
                         const words = d.split(':::');
                         const text = words[0];
                         var size = 1;
                         if (words.length > 1) {
                             size = parseFloat(words[1]);
                             size = Math.log(size) * 15;
                         }
                         return {text : words[0], size : size};
                     }))
                     .padding(2)
                     .rotate(function() {
                         return 0;
                         ~~(Math.random() * 2) * 90 - 45;
                     })
                     .font("Impact")
                     .fontSize(function(d) { return d.size; })
                     .on("end", draw);

    layout.start();

    function draw(words) {
        d3.select("#word-cloud")
            .append("svg")
            .attr("width", layout.size()[0])
            .attr("height", layout.size()[1])
            .append("g")
            .attr("transform", "translate(" + layout.size()[0] / 2 + "," +
                                   layout.size()[1] / 2 + ")")
            .selectAll("text")
            .data(words)
            .enter()
            .append("text")
            .style("font-size", function(d) { return d.size + "px"; })
            .style("font-family", "Impact")
            .attr("text-anchor", "middle")
            .attr("transform",
                  function(d) {
                      return "translate(" + [ d.x, d.y ] + ")rotate(" +
                             d.rotate + ")";
                  })
            .text(function(d) { return d.text; });

        d3.selectAll("text").style("fill", function() { return '#999999'; });

        d3.selectAll("g text").on("click", function() {
            const current_word = d3.select(this).text();
            if (highlighted_words.includes(current_word)) {
                highlighted_words.remove(current_word);
            } else {
                highlighted_words.push(current_word);
            }

            d3.selectAll("text").style("fill", function() { 
                for (const w of highlighted_words) {
                    if (w==d3.select(this).text()) {
                        return '#FF6341';
                    }
                }
                return '#999999'; 
            });
            const logic = d3.select('input[name="word-cloud-logic"]:checked').node().value;
            d3.selectAll("div.paper").style('background-color', function() {
                if (logic=='or') {
                    for (const w of highlighted_words) {
                        if (d3.select(this).text().indexOf(w) != -1) {
                            return '#FFC0B2';
                        }
                    }
                    return '';
                } else {
                    for (const w of highlighted_words) {
                        if (d3.select(this).text().indexOf(w) == -1) {
                            return '';
                        }
                    }
                    return '#FFC0B2';
                }
            });
        });
    }
});