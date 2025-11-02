const fs = require('fs');
const request = require('request');
const https = require('https');

[
  "Cheese",
  "-Beach",
  "-Desert",
  "-Farm",
  "-Forest",
  "-Hell",
  "Art",
  "Construction",
  "Desert",
  "Dungeon",
  "Easter",
  "Forest",
  "Fruit",
  "Gulf",
  "Hell",
  "Hospital",
  "Jungle",
  "Manhattan",
  "Medieval",
  "Music",
  "Pirate",
  "Snow",
  "Space",
  "Sports",
  "Tentacle",
  "Time",
  "Tools",
  "Tribal",
  "Urban"
].map(async t => {
  for (let i = 1; i < 6; i++) {
    const html = await new Promise((resolve, _) => {
      setTimeout(() => {
        const htmlUrl = `https://cwt.normalnonoobs.com/maps/${i}/${t}`;
        console.log(htmlUrl);
        https.get(htmlUrl, res => {
          let buff = '';
          res.on('data', d => buff += d);
          res.on('end', () => resolve(buff));
        });
      }, Math.random() * 4 + 1);
    });
    html.matchAll(/src="(\/assets\/map\/.*\.png)"/g)
      .map(m => m[1])
      .forEach(async m => {
        console.log(m);
        const mapUrl = 'https://cwt.normalnonoobs.com' + m;
        await new Promise((resolve, reject) => {
          setTimeout(() => {
            request.head(mapUrl, function (err, res, body) {
              request(mapUrl)
                .pipe(fs.createWriteStream('target/' + t + '_' + m.split("/").slice(-1)[0]))
                .on('close', resolve);
            });
          }, Math.random() * 4 + 1);
        });
      });
  }
})

