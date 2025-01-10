const express = require('express');
const path = require('path');
const fileUpload = require('express-fileupload');

const app = express();
const indexRoutes = require('./routes/index');

// Middleware
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.urlencoded({ extended: true }));
app.use(fileUpload());
app.set('view engine', 'ejs');
app.engine('ejs', require('ejs-mate'));
app.set('views', path.join(__dirname, 'views'));

// Routes
app.use('/', indexRoutes);
app.get("/options",(req,res)=>{
    res.render("options");
})
app.get("/about",(req,res)=>{
    res.render("about");
})
// Start Server
const PORT = 8080;
app.listen(PORT, () => console.log(`Server is running on http://localhost:${PORT}`));
