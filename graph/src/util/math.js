
import _ from 'underscore';
import kmath from 'kmath';

const knumber = kmath.number;

export const KhanMath = {
    // Simplify formulas before display
    cleanMath: function(expr) {
        return typeof expr === "string" ?
            expr.replace(/\+\s*-/g, "- ")
                .replace(/-\s*-/g, "+ ")
                .replace(/\^1/g, "") :
            expr;
    },

    // Bound a number by 1e-6 and 1e20 to avoid exponents after toString
    bound: function(num) {
        if (num === 0) {
            return num;
        } else if (num < 0) {
            return -KhanMath.bound(-num);
        } else {
            return Math.max(1e-6, Math.min(num, 1e20));
        }
    },
    //求阶乘
    factorial: function(x) {
        if (x <= 1) {
            return x;
        } else {
            return x * KhanMath.factorial(x - 1);
        }
    },

    //求最大公约数
    getGCD: function(a, b) {
        if (arguments.length > 2) {
            const rest = [].slice.call(arguments, 1);
            return KhanMath.getGCD(a, KhanMath.getGCD(...rest));
        } else {
            let mod;

            a = Math.abs(a);
            b = Math.abs(b);

            while (b) {
                mod = a % b;
                a = b;
                b = mod;
            }

            return a;
        }
    },

    getLCM: function(a, b) {
        if (arguments.length > 2) {
            const rest = [].slice.call(arguments, 1);
            return KhanMath.getLCM(a, KhanMath.getLCM(...rest));
        } else {
            return Math.abs(a * b) / KhanMath.getGCD(a, b);
        }
    },

    primes: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
        47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97],

    //判断一个数是否是素数
    isPrime: function(n) {
        if (n <= 1) {
            return false;
        } else if (n < 101) {
            return !!$.grep(KhanMath.primes, function(p, i) {
                return Math.abs(p - n) <= 0.5;
            }).length;
        } else {
            if (n <= 1 || n > 2 && n % 2 === 0) {
                return false;
            } else {
                for (let i = 3, sqrt = Math.sqrt(n); i <= sqrt; i += 2) {
                    if (n % i === 0) {
                        return false;
                    }
                }
            }

            return true;
        }

    },

    //求一个数的质因数
    getPrimeFactorization: function(number) {
        if (number === 1) {
            return [];
        } else if (KhanMath.isPrime(number)) {
            return [number];
        }

        const maxf = Math.sqrt(number);
        for (let f = 2; f <= maxf; f++) {
            if (number % f === 0) {
                return $.merge(
                    KhanMath.getPrimeFactorization(f),
                    KhanMath.getPrimeFactorization(number / f)
                );
            }
        }
    },

    // Round a number to the nearest increment
    // E.g., if increment = 30 and num = 40, return 30. if increment = 30 and
    //     num = 45, return 60.
    roundToNearest: function(increment, num) {
        return Math.round(num / increment) * increment;
    },

    // Round a number to a certain number of decimal places
    roundTo: function(precision, num) {
        const factor = Math.pow(10, precision).toFixed(5);
        return Math.round((num * factor).toFixed(5)) / factor;
    },

    /**
     * Return a string of num rounded to a fixed precision decimal places,
     * with an approx symbol if num had to be rounded, and trailing 0s
     */
    toFixedApprox: function(num, precision) {
        // TODO(aria): Make this locale-dependent like KhanUtil.localeToFixed
        const fixedStr = num.toFixed(precision);
        if (knumber.equal(+fixedStr, num)) {
            return fixedStr;
        } else {
            return "\\approx " + fixedStr;
        }
    },

    /**
     * Return a string of num rounded to precision decimal places, with an
     * approx symbol if num had to be rounded, but no trailing 0s if it was
     * not rounded.
     */
    roundToApprox: function(num, precision) {
        const fixed = KhanMath.roundTo(precision, num);
        if (knumber.equal(fixed, num)) {
            return String(fixed);
        } else {
            return KhanMath.toFixedApprox(num, precision);
        }
    },

    // toFraction(4/8) => [1, 2]
    // toFraction(0.666) => [333, 500]
    // toFraction(0.666, 0.001) => [2, 3]
    //
    // tolerance can't be bigger than 1, sorry
    //化简成最简分数
    toFraction: function(decimal, tolerance) {
        if (tolerance == null) {
            tolerance = Math.pow(2, -46);
        }

        if (decimal < 0 || decimal > 1) {
            let fract = decimal % 1;
            fract += (fract < 0 ? 1 : 0);

            const nd = KhanMath.toFraction(fract, tolerance);
            nd[0] += Math.round(decimal - fract) * nd[1];
            return nd;
        } else if (Math.abs(Math.round(Number(decimal)) - decimal) <=
                tolerance) {
            return [Math.round(decimal), 1];
        } else {
            let loN = 0;
            let loD = 1;
            let hiN = 1;
            let hiD = 1;
            let midN = 1;
            let midD = 2;

            while (true) { // @Nolint(constant condition)
                if (Math.abs(Number(midN / midD) - decimal) <= tolerance) {
                    return [midN, midD];
                } else if (midN / midD < decimal) {
                    loN = midN;
                    loD = midD;
                } else {
                    hiN = midN;
                    hiD = midD;
                }

                midN = loN + hiN;
                midD = loD + hiD;
            }
        }
    },


    // 给定一个数，通过正则匹配 返回数的类型，如整数，分数，混分数， pi
    getNumericFormat: function(text) {
        text = $.trim(text);
        text = text.replace(/\u2212/, "-").replace(/([+-])\s+/g, "$1");
        if (text.match(/^[+-]?\d+$/)) {
            return "integer";
        } else if (text.match(/^[+-]?\d+\s+\d+\s*\/\s*\d+$/)) {
            return "mixed";
        }
        const fraction = text.match(/^[+-]?(\d+)\s*\/\s*(\d+)$/);
        if (fraction) {
            return parseFloat(fraction[1]) > parseFloat(fraction[2]) ?
                    "improper" : "proper";
        } else if (text.replace(/[,. ]/g, "").match(/^\d+$/)) {
            return "decimal";
        } else if (text.match(/(pi?|\u03c0|t(?:au)?|\u03c4|pau)/)) {
            return "pi";
        } else {
            return null;
        }
    },


    // Returns a string of the number in a specified format
    toNumericString: function(number, format) {
        if (number == null) {
            return "";
        } else if (number === 0) {
            return "0"; // otherwise it might end up as 0% or 0pi
        }

        if (format === "percent") {
            return number * 100 + "%";
        }

        if (format === "pi") {
            const fraction = knumber.toFraction(number / Math.PI);
            const numerator = Math.abs(fraction[0]);
            const denominator = fraction[1];
            if (knumber.isInteger(numerator)) {
                const sign = number < 0 ? "-" : "";
                const pi = "\u03C0";
                return sign + (numerator === 1 ? "" : numerator) + pi +
                    (denominator === 1 ? "" : "/" + denominator);
            }
        }

        if (_(["proper", "improper", "mixed", "fraction"]).contains(format)) {
            const fraction = knumber.toFraction(number);
            const numerator = Math.abs(fraction[0]);
            const denominator = fraction[1];
            const sign = number < 0 ? "-" : "";
            if (denominator === 1) {
                return sign + numerator; // for integers, irrational, d > 1000
            } else if (format === "mixed") {
                const modulus = numerator % denominator;
                const integer = (numerator - modulus) / denominator;
                return sign + (integer ? integer + " " : "") +
                        modulus + "/" + denominator;
            } // otherwise proper, improper, or fraction
            return sign + numerator + "/" + denominator;
        }

        // otherwise (decimal, float, long long)
        return String(number);
    },
};

function findChildOrAdd(elem, className) {
    const $child = $(elem).find("." + className);
    if ($child.length === 0) {
        return $("<span>").addClass(className).appendTo($(elem));
    } else {
        return $child;
    }
}

function doCallback(elem, callback) {
    let tries = 0;
    (function check() {
        const height = elem.scrollHeight;
        // Heuristic to guess if the font has kicked in
        // so we have box metrics (magic number ick,
        // but this seems to work mostly-consistently)
        if (height > 18 || tries >= 10) {
            callback();
        } else {
            tries++;
            setTimeout(check, 100);
        }
    })();
}

export const TexUtil = {
    // Process a node and add math inside of it. This attempts to use KaTeX to
    // format the math, and if that fails it falls back to MathJax.
    //
    // elem: The element which the math should be added to.
    //
    // text: The text that should be formatted inside of the node. If the node
    //       has already had math formatted inside of it before, this doesn't
    //       have to be provided. If this is not provided, and the node hasn't
    //       been formatted before, the text content of the node is used.
    //
    // force: (optional) if the node has been processed before, then it will
    //        not be formatted again, unless this argument is true
    //
    // callback: (optional) a callback to be run after the math has been
    //           processed (note: this might be called synchronously or
    //           asynchronously, depending on whether KaTeX or MathJax is used)
    processMath: function(elem, text, force, callback) {
        const $elem = $(elem);

        // Only process if it hasn't been done before, or it is forced
        if ($elem.attr("data-math-formula") == null || force) {
            const $katexHolder = findChildOrAdd($elem, "katex-holder");
            const $mathjaxHolder = findChildOrAdd($elem, "mathjax-holder");

            // Search for MathJax-y script tags inside of the node. These are
            // used by MathJax to denote the formula to be typeset. Before, we
            // would update the formula by updating the contents of the script
            // tag, which shouldn't happen any more, but we manage them just in
            // case.
            const script = $mathjaxHolder.find("script[type='math/tex']")[0];

            // If text wasn't provided, we look in two places
            if (text == null) {
                if ($elem.attr("data-math-formula")) {
                    // The old typeset formula
                    text = $elem.attr("data-math-formula");
                } else if (script) {
                    // The contents of the <script> tag
                    text = script.text || script.textContent;
                }
            }

            text = text != null ? text + "" : "";

            // Attempt to clean up some of the math
            text = KhanMath.cleanMath(text);

            // Store the formula that we're using
            $elem.attr("data-math-formula", text);

            // Try to process the nodes with KaTeX first
            try {
                katex.render(text, $katexHolder[0]);
                // If that worked, and we previously formatted with
                // mathjax, do some mathjax cleanup
                if ($elem.attr("data-math-type") === "mathjax") {
                    // Remove the old mathjax stuff
                    const jax = MathJax.Hub.getJaxFor(script);
                    if (jax) {
                        const e = jax.SourceElement();
                        if (e.previousSibling &&
                            e.previousSibling.className) {
                            jax.Remove();
                        }
                    }
                }
                $elem.attr("data-math-type", "katex");
                // Call the callback
                if (callback) {
                    doCallback(elem, callback);
                }
                return;
            } catch (err) {
                // IE doesn't do instanceof correctly, so we resort to
                // manual checking
                /* jshint -W103 */
                if (err.__proto__ !== katex.ParseError.prototype) {
                    throw err;
                }
                /* jshint +W103 */
            }

            // Otherwise, fallback to MathJax

            // (Note: we don't need to do any katex cleanup here, because
            // KaTeX is smart and cleans itself up)
            $elem.attr("data-math-type", "mathjax");
            // Update the script tag, or add one if necessary
            if (!script) {
                $mathjaxHolder.append("<script type='math/tex'>" +
                    text.replace(/<\//g, "< /") + "</script>");
            } else {
                if ("text" in script) {
                    // IE8, etc
                    script.text = text;
                } else {
                    script.textContent = text;
                }
            }
            if (typeof MathJax !== "undefined") {
                // Put the process, a debug log, and the callback into the
                // MathJax queue
                MathJax.Hub.Queue(["Reprocess", MathJax.Hub,
                    $mathjaxHolder[0]]);
                MathJax.Hub.Queue(function() {
                    KhanUtil.debugLog("MathJax done typesetting (" + text +
                        ")");
                });
                if (callback) {
                    MathJax.Hub.Queue(function() {
                        const cb = MathJax.Callback(function() {});
                        doCallback(elem, function() {
                            callback();
                            cb();
                        });
                        return cb;
                    });
                }
            }
        }
    },

    processAllMath: function(elem, force) {
        const $elem = $(elem);
        $elem.filter("code").add($elem.find("code")).each(function() {
            const $this = $(this);
            let text = $this.attr("data-math-formula");
            if (text == null) {
                text = $this.text();
                $this.empty();
            }
            KhanUtil.processMath(this, text, force);
        });
    },

    // Function to restore a node to a non-math-processed state
    cleanupMath: function(elem) {
        const $elem = $(elem);

        // Only mess with it if it's been processed before
        if ($elem.attr("data-math-formula")) {
            // Remove MathJax remnants
            if (typeof MathJax !== "undefined") {
                const jax = MathJax.Hub.getJaxFor($elem.find("script")[0]);
                if (jax) {
                    const e = jax.SourceElement();
                    if (e.previousSibling && e.previousSibling.className) {
                        jax.Remove();
                    }
                }
            }

            $elem.text($elem.attr("data-math-formula"));
            $elem.attr("data-math-formula", null);
            $elem.attr("data-math-type", null);
        }

        return elem;
    }
}

