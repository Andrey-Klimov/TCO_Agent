const API_URL = 'https://tocagentv2-production.up.railway.app/'; // замените при деплое

// Справочник регионов РФ
const REGION_NAMES = {
    '1': 'Республика Адыгея', '2': 'Республика Башкортостан', '3': 'Республика Бурятия',
    '4': 'Республика Алтай', '5': 'Республика Дагестан', '6': 'Республика Ингушетия',
    '7': 'Кабардино-Балкария', '8': 'Республика Калмыкия', '9': 'Карачаево-Черкесия',
    '10': 'Республика Карелия', '11': 'Республика Коми', '12': 'Республика Марий Эл',
    '13': 'Республика Мордовия', '14': 'Республика Саха (Якутия)', '15': 'Северная Осетия',
    '16': 'Республика Татарстан', '17': 'Республика Тыва', '18': 'Удмуртия',
    '19': 'Республика Хакасия', '20': 'Чеченская Республика', '21': 'Чувашия',
    '22': 'Алтайский край', '23': 'Краснодарский край', '24': 'Красноярский край',
    '25': 'Приморский край', '26': 'Ставропольский край', '27': 'Хабаровский край',
    '28': 'Амурская область', '29': 'Архангельская область', '30': 'Астраханская область',
    '31': 'Белгородская область', '32': 'Брянская область', '33': 'Владимирская область',
    '34': 'Волгоградская область', '35': 'Вологодская область', '36': 'Воронежская область',
    '37': 'Ивановская область', '38': 'Иркутская область', '39': 'Калининградская область',
    '40': 'Калужская область', '41': 'Камчатский край', '42': 'Кемеровская область',
    '43': 'Кировская область', '44': 'Костромская область', '45': 'Курганская область',
    '46': 'Курская область', '47': 'Ленинградская область', '48': 'Липецкая область',
    '49': 'Магаданская область', '50': 'Московская область', '51': 'Мурманская область',
    '52': 'Нижегородская область', '53': 'Новгородская область', '54': 'Новосибирская область',
    '55': 'Омская область', '56': 'Оренбургская область', '57': 'Орловская область',
    '58': 'Пензенская область', '59': 'Пермский край', '60': 'Псковская область',
    '61': 'Ростовская область', '62': 'Рязанская область', '63': 'Самарская область',
    '64': 'Саратовская область', '65': 'Сахалинская область', '66': 'Свердловская область',
    '67': 'Смоленская область', '68': 'Тамбовская область', '69': 'Тверская область',
    '70': 'Томская область', '71': 'Тульская область', '72': 'Тюменская область',
    '73': 'Ульяновская область', '74': 'Челябинская область', '75': 'Забайкальский край',
    '76': 'Ярославская область', '77': 'г. Москва', '78': 'г. Санкт-Петербург',
    '79': 'Еврейская АО', '83': 'Ненецкий АО', '86': 'Ханты-Мансийский АО',
    '87': 'Чукотский АО', '89': 'Ямало-Ненецкий АО', '91': 'Республика Крым',
    '92': 'г. Севастополь',
};

function getRegionLabel(code) {
    const name = REGION_NAMES[code];
    return name ? `${code} — ${name}` : `Регион ${code}`;
}

// ---------- Кастомный селект регионов ----------
const regionSearch = document.getElementById('regionSearch');
const regionValue = document.getElementById('regionValue');
const regionOptions = document.getElementById('regionOptions');
let allRegions = [];
let filteredRegions = [];

function loadRegions() {
    fetch(`${API_URL}/regions`)
        .then(res => res.json())
        .then(regions => {
            const unique = [...new Set(regions.map(r => String(r)))];
            allRegions = unique.sort((a,b) => {
                const na = parseInt(a), nb = parseInt(b);
                if (!isNaN(na) && !isNaN(nb)) return na - nb;
                return a.localeCompare(b);
            });
            filteredRegions = [...allRegions];
            renderOptions(filteredRegions);
        })
        .catch(() => { allRegions = []; filteredRegions = []; });
}

function renderOptions(list) {
    regionOptions.innerHTML = '';
    const allItem = document.createElement('li');
    allItem.textContent = 'Все регионы';
    allItem.dataset.value = '';
    allItem.addEventListener('click', () => selectRegion(''));
    regionOptions.appendChild(allItem);
    if (list.length === 0) {
        const no = document.createElement('li');
        no.textContent = 'Нет регионов';
        no.classList.add('no-results');
        regionOptions.appendChild(no);
        return;
    }
    list.forEach(code => {
        const li = document.createElement('li');
        li.textContent = getRegionLabel(code);
        li.dataset.value = code;
        li.addEventListener('click', () => selectRegion(code));
        regionOptions.appendChild(li);
    });
}

function selectRegion(code) {
    regionValue.value = code;
    regionSearch.value = code ? getRegionLabel(code) : 'Все регионы';
    regionOptions.style.display = 'none';
}

regionSearch.addEventListener('focus', () => {
    const term = regionSearch.value.trim().toLowerCase();
    filteredRegions = (term === '' || term === 'все регионы') ? [...allRegions] :
        allRegions.filter(code => getRegionLabel(code).toLowerCase().includes(term));
    renderOptions(filteredRegions);
    regionOptions.style.display = 'block';
});

regionSearch.addEventListener('input', () => {
    const term = regionSearch.value.trim().toLowerCase();
    filteredRegions = allRegions.filter(code => getRegionLabel(code).toLowerCase().includes(term));
    renderOptions(filteredRegions);
    regionOptions.style.display = 'block';
});

document.addEventListener('click', (e) => {
    if (!document.getElementById('regionGroup').contains(e.target)) {
        regionOptions.style.display = 'none';
    }
});

loadRegions();

// ---------- Утилиты ----------
const getFloat = v => { const n = parseFloat(v); return isNaN(n) ? null : n; };
const getInt = v => { const n = parseInt(v); return isNaN(n) ? null : n; };
function getRegionValue() { return regionValue.value || null; }

// ---------- Состояние и сортировка ----------
let lastResults = [];
let currentSortIndex = 0;

const sortRange = document.getElementById('sortRange');
sortRange.addEventListener('input', () => {
    currentSortIndex = parseInt(sortRange.value);
    renderResults(lastResults);
});

function sortResults(results) {
    const sorted = [...results];
    if (currentSortIndex === 0) sorted.sort((a,b) => b.final_score - a.final_score);
    else if (currentSortIndex === 1) sorted.sort((a,b) => a.tco_pred_ths - b.tco_pred_ths);
    else sorted.sort((a,b) => b.tco_pred_ths - a.tco_pred_ths);
    return sorted;
}

function calculateOverallRating(results) {
    if (!results.length) return 0;
    const sum = results.reduce((acc, obj) => acc + obj.final_score, 0);
    return Math.round((sum / results.length) * 100);
}

function renderResults(results) {
    const container = document.getElementById('resultsContainer');
    const empty = document.getElementById('emptyResult');
    const sortPanel = document.getElementById('sortPanel');
    const widget = document.getElementById('overallRatingWidget');
    const infoCard = document.getElementById('infoCard');               ///

    if (!results || results.length === 0) {
        container.innerHTML = '';
        empty.style.display = 'block';
        sortPanel.style.display = 'none';
        widget.style.display = 'none';
        infoCard.style.display = 'none';                               ///
        return;
    }
    empty.style.display = 'none';
    sortPanel.style.display = 'flex';
    widget.style.display = 'block';
    infoCard.style.display = 'flex';                                   ///

    const overall = calculateOverallRating(results);
    document.getElementById('overallScore').textContent = overall;

    const sorted = sortResults(results);
    let html = '';
    sorted.forEach(obj => {
        const area = obj.obsh_plosh.toFixed(1);
        const seg = obj.segmentt ? obj.segmentt : 'Объект';
        const reg = obj.region || '?';
        const title = `${seg}, ${area} м², регион ${reg}`;
        const firstLineBadge = obj.first_line == 1
            ? '<span class="first-line-badge"><i class="fas fa-check-circle"></i> 1-я линия</span>'
            : '';
        html += `
          <div class="object-card">
            <div class="object-title">${title} ${firstLineBadge}</div>
            <div class="object-meta">
              <span><i class="fas fa-vector-square"></i> ${area} м²</span>
              <span><i class="fas fa-user-tie"></i> ${obj.quan} мест</span>
              <span><i class="fas fa-calendar-alt"></i> ${obj.building_age.toFixed(1)} лет</span>
            </div>
            <div class="object-price">
              <i class="fas fa-ruble-sign"></i> TCO: ${obj.tco_pred_ths.toFixed(1)} тыс. руб.
            </div>
          </div>
        `;
    });
    container.innerHTML = html;
}

// ---------- Обработчик поиска ----------
document.getElementById('searchBtn').addEventListener('click', () => {
    const mode = document.querySelector('input[name="mode"]:checked')?.value || 'own';
    const region = getRegionValue();
    const areaMin = getFloat(document.getElementById('area_min').value);
    const areaMax = getFloat(document.getElementById('area_max').value);
    const quanMin = getInt(document.getElementById('quan_min').value);
    const quanMax = getInt(document.getElementById('quan_max').value);
    const ageMin = getFloat(document.getElementById('age_min').value);
    const ageMax = getFloat(document.getElementById('age_max').value);
    const firstLine = document.getElementById('first_line').checked ? 1 : 0;
    const tcoMin = getFloat(document.getElementById('tco_min').value);
    const tcoMax = getFloat(document.getElementById('tco_max').value);

    const payload = {
        mode,
        region,
        obsh_plosh_min: areaMin, obsh_plosh_max: areaMax,
        quan_min: quanMin, quan_max: quanMax,
        building_age_min: ageMin, building_age_max: ageMax,
        first_line: firstLine,
        tco_min: tcoMin, tco_max: tcoMax,
        top_k: 10
    };

    // Сброс и индикация загрузки
    document.getElementById('resultsContainer').innerHTML = '';
    document.getElementById('emptyResult').style.display = 'none';
    document.getElementById('sortPanel').style.display = 'none';
    document.getElementById('overallRatingWidget').style.display = 'none';
    document.getElementById('infoCard').style.display = 'none';                    ///
    document.getElementById('loadingIndicator').style.display = 'block';

    fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('loadingIndicator').style.display = 'none';
        if (data.error) {
            document.getElementById('resultsContainer').innerHTML =
                `<div class="empty-state"><p style="color:red;">${data.error}</p></div>`;
            return;
        }
        lastResults = Array.isArray(data) ? data : [];
        sortRange.value = 0;
        currentSortIndex = 0;
        renderResults(lastResults);
    })
    .catch(err => {
        document.getElementById('loadingIndicator').style.display = 'none';
        document.getElementById('resultsContainer').innerHTML =
            `<div class="empty-state"><p style="color:red;">Ошибка сети: ${err.message}</p></div>`;
    });
});