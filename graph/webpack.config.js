/**
 * Created by wangwei on 17/6/22.
 */

var path = require('path');
var webpack = require('webpack');
// var htmlWebpackPlugin =  require('html-webpack-plugin');

module.exports = {
    entry: [
        'webpack/hot/only-dev-server',
        path.resolve(__dirname, './src/index.js')
    ],
    output: {
        path: path.resolve(__dirname, './build'),
        filename: "bundle.js"
    },
    module: {
        loaders: [
            {
                test: /\.js$/,
                exclude: /node_modules/,
                loader: "babel-loader",
                query:
                    {
                        presets:['react','es2015']
                    }
            },
            {
                test: /\.css$/,
                loader: 'style-loader!css-loader'
            },

            {
                test: /\.less$/,
                loader: 'style!css!less'
            },

            {
                test: /\.(png|jpg)$/,
                loader: 'url?limit=40000'
            }
        ]
    },
    // resolve:{
    //     extensions:['','.js','.json']
    // },

    devServer: {
        inline: true,
        port: 8181
    },

    externals:{
        "jquery": "jQuery"
    },
    // devServer: {
    //     hot: true,
    //     inline: true
    // },
    plugins: [
        new webpack.NoErrorsPlugin(),
        new webpack.HotModuleReplacementPlugin()
    ]
};