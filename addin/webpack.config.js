const path = require("path");
const CopyWebpackPlugin = require("copy-webpack-plugin");

module.exports = {
  mode: "development",
  devServer: {
    static: {
      directory: path.join(__dirname, "dist"),
    },
    compress: true,
    port: 3000,
    server: {
      type: "https", // ✅ moved here
    },
    hot: false,
    liveReload: true,
  },
  plugins: [
    new CopyWebpackPlugin({
      patterns: [
        { from: "taskpane.html", to: "taskpane.html" },
        { from: "function-file.html", to: "function-file.html" },
        { from: "assets", to: "assets" },
      ],
    }),
  ],
  entry: {
    taskpane: "./taskpane.js",
    functionFile: "./function-file.js",
  },
  output: {
    filename: "[name].bundle.js",
    path: path.resolve(__dirname, "dist"),
  },
};
