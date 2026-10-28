"use client";

import { ChangeEvent, DragEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import NextImage from "next/image";
import {
  Bot, Bug, CalendarDays, Camera, ChevronRight, CloudDrizzle, CloudRain, CloudSun, Crosshair,
  Droplets, FileWarning, Gauge, ImagePlus, Leaf, LoaderCircle, Map, MapPin, Menu, MessageCircle,
  Mic, MicOff, MoveUpRight, Navigation, ScanLine, Send, ShieldCheck, ShoppingBasket, Sparkles,
  Sprout, ThermometerSun, TrendingDown, TrendingUp, Upload, Wind, X,
} from "lucide-react";
import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  cropGuides, createForecast, createPriceHistory, crops, districts, formatBnNumber, nearestDistrict,
} from "@/lib/agri-data";

type Screen = "dashboard" | "market" | "forecast" | "weather" | "disease" | "chat" | "advice";
type Coordinates = { lat: number; lng: number; source: "district" | "device" };
type WeatherData = {
  weather: {
    current: { temperature_2m: number; relative_humidity_2m: number; apparent_temperature: number; precipitation: number; weather_code: number; wind_speed_10m: number };
    daily: { time: string[]; weather_code: number[]; temperature_2m_max: number[]; temperature_2m_min: number[]; precipitation_probability_max: number[]; precipitation_sum: number[]; wind_speed_10m_max: number[] };
  };
  source: string;
  fetchedAt: string;
};
type ChatMessage = { role: "user" | "assistant"; content: string };
type SpeechAlternativeLike = { transcript: string };
type SpeechResultLike = ArrayLike<SpeechAlternativeLike> & { isFinal: boolean };
type SpeechResultEventLike = Event & { resultIndex: number; results: ArrayLike<SpeechResultLike> };
type SpeechErrorEventLike = Event & { error: string };
type SpeechRecognitionLike = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((event: SpeechResultEventLike) => void) | null;
  onerror: ((event: SpeechErrorEventLike) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};
type SpeechRecognitionWindow = Window & {
  SpeechRecognition?: new () => SpeechRecognitionLike;
  webkitSpeechRecognition?: new () => SpeechRecognitionLike;
};
type DiseasePrediction = { class_label: string; crop_label: string; label: string; score_percent: number };
type DiseaseProduct = {
  name: string;
  registration_no: string;
  active_ingredients: string;
  label_scope: string;
  rate: string;
  application_method: string;
  directions: string[];
  source_label: string;
  source_url: string;
};
type DiseaseDiagnosis = {
  source: "krittim_vision" | "plantvillage_classifier";
  leaf_status: "leaf" | "not_leaf" | "uncertain";
  crop_label: string;
  disease_label: string;
  cause_type: string;
  cause_label_bn: string;
  visible_signs_bn: string[];
  nonchemical_actions_bn: string[];
  confidence_band: "low" | "medium" | "high";
};
type DiseaseGuidance = {
  category: string;
  reason: string;
  signs_to_check: string;
  actions: string[];
  product_note: string;
  products: DiseaseProduct[];
  watering: { care: string; amount_note: string };
  source_checked_on: string;
  sources: Array<{ label: string; url: string }>;
};
type DiseaseResponse = {
  prediction: DiseasePrediction;
  diagnosis: DiseaseDiagnosis;
  top_predictions: DiseasePrediction[];
  scope_warning: string;
  guidance: DiseaseGuidance;
  vision_analysis: { available: boolean; status: string };
  local_model_available: boolean;
  image_sent_to_vision_service: boolean;
  summary?: string | null;
  summary_requested: boolean;
  summary_available: boolean;
  summary_error_code?: string | null;
};
type PricePoint = { day: string; price: number };
type PriceApiResponse = {
  history: PricePoint[];
  forecast: PricePoint[];
  source: string;
  sample_data_as_of?: string;
  data_kind?: string;
  note?: string;
};

async function fileToJpegDataUrl(file: File): Promise<string> {
  const objectUrl = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = objectUrl;
    try {
      await image.decode();
    } catch {
      throw new Error("এই ছবিটি ব্রাউজারে পড়া যাচ্ছে না। JPG, PNG বা WebP ছবিতে আবার চেষ্টা করুন।");
    }
    if (!image.naturalWidth || !image.naturalHeight || image.naturalWidth * image.naturalHeight > 24_000_000) {
      throw new Error("ছবির রেজোলিউশন কমিয়ে আবার চেষ্টা করুন।");
    }
    let maxDimension = 1280;
    let quality = 0.86;
    let blob: Blob | null = null;
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    if (!context) throw new Error("ছবি প্রস্তুত করা যাচ্ছে না।");

    for (let attempt = 0; attempt < 5; attempt += 1) {
      const scale = Math.min(1, maxDimension / Math.max(image.naturalWidth, image.naturalHeight));
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
      if (blob && blob.size <= 1_850_000) break;
      maxDimension = Math.max(560, Math.round(maxDimension * 0.8));
      quality = Math.max(0.55, quality - 0.07);
    }
    if (!blob || blob.size > 2 * 1024 * 1024) throw new Error("ছবিটি ছোট করে আবার চেষ্টা করুন।");
    return await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => typeof reader.result === "string" ? resolve(reader.result) : reject(new Error("ছবির তথ্য পড়া যায়নি।"));
      reader.onerror = () => reject(new Error("ছবির তথ্য পড়া যায়নি।"));
      reader.readAsDataURL(blob);
    });
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

const navigation: Array<{ id: Screen; label: string; icon: typeof Sprout }> = [
  { id: "dashboard", label: "ড্যাশবোর্ড", icon: Sprout },
  { id: "market", label: "আজকের বাজারদর", icon: ShoppingBasket },
  { id: "forecast", label: "মূল্য পূর্বাভাস", icon: TrendingUp },
  { id: "weather", label: "আবহাওয়া", icon: CloudSun },
  { id: "disease", label: "রোগ সহায়তা", icon: ScanLine },
  { id: "chat", label: "AI কৃষি সহায়ক", icon: MessageCircle },
  { id: "advice", label: "ফসল পরামর্শ", icon: Leaf },
];

const wmo = (code: number) => {
  if ([0, 1].includes(code)) return { label: "পরিষ্কার আকাশ", icon: CloudSun, color: "sun" };
  if ([2, 3, 45, 48].includes(code)) return { label: "আংশিক মেঘলা", icon: CloudSun, color: "cloud" };
  if ([51, 53, 55, 56, 57].includes(code)) return { label: "গুঁড়ি গুঁড়ি বৃষ্টি", icon: CloudDrizzle, color: "rain" };
  if ([61, 63, 65, 66, 67, 80, 81, 82].includes(code)) return { label: "বৃষ্টি", icon: CloudRain, color: "rain" };
  if ([95, 96, 99].includes(code)) return { label: "বজ্রসহ বৃষ্টি", icon: CloudRain, color: "storm" };
  return { label: "পরিবর্তনশীল", icon: CloudSun, color: "cloud" };
};

const banglaDate = (value: string, weekday = false) => new Intl.DateTimeFormat("bn-BD", {
  weekday: weekday ? "short" : undefined, day: "numeric", month: "short",
}).format(new Date(`${value}T12:00:00`));

const mapUrl = (lat: number, lng: number) => {
  const offset = 0.11;
  const bbox = `${lng - offset}%2C${lat - offset}%2C${lng + offset}%2C${lat + offset}`;
  return `https://www.openstreetmap.org/export/embed.html?bbox=${bbox}&layer=mapnik&marker=${lat}%2C${lng}`;
};

export default function HomePage() {
  const [active, setActive] = useState<Screen>("dashboard");
  const [districtName, setDistrictName] = useState("ঢাকা");
  const [crop, setCrop] = useState("ধান");
  const initialDistrict = districts.find((item) => item.name === "ঢাকা") ?? districts[0];
  const [coordinates, setCoordinates] = useState<Coordinates>({ lat: initialDistrict.lat, lng: initialDistrict.lng, source: "district" });
  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [weatherError, setWeatherError] = useState("");
  const [weatherLoading, setWeatherLoading] = useState(true);
  const [locating, setLocating] = useState(false);
  const [locationMessage, setLocationMessage] = useState("জেলা নির্বাচন করে বা নিজের অবস্থান থেকে পূর্বাভাস দেখুন");
  const [mobileNav, setMobileNav] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    { role: "assistant", content: "আসসালামু আলাইকুম। আমি স্মার্ট কৃষি বুদ্ধিমত্তা। ফসল, আবহাওয়া বা পরিচর্যা নিয়ে প্রশ্ন করতে পারেন।" },
  ]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");
  const [plantImage, setPlantImage] = useState<string | null>(null);
  const [plantFile, setPlantFile] = useState<File | null>(null);
  const [plantFileName, setPlantFileName] = useState("");
  const [symptom, setSymptom] = useState("");
  const [diseaseResult, setDiseaseResult] = useState<DiseaseResponse | null>(null);
  const [diseaseError, setDiseaseError] = useState("");
  const [diseaseLoading, setDiseaseLoading] = useState(false);
  const [includeAiExplanation, setIncludeAiExplanation] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [priceData, setPriceData] = useState<PriceApiResponse | null>(null);
  const [priceLoading, setPriceLoading] = useState(true);
  const galleryInput = useRef<HTMLInputElement>(null);
  const cameraInput = useRef<HTMLInputElement>(null);
  const diseaseAbort = useRef<AbortController | null>(null);
  const diseaseRequestId = useRef(0);

  const localHistory = useMemo(() => createPriceHistory(crop, districtName), [crop, districtName]);
  const localForecast = useMemo(() => createForecast(crop, districtName), [crop, districtName]);
  const history = priceData?.history ?? localHistory;
  const forecast = priceData?.forecast ?? localForecast;
  const forecastSource = priceData?.source ?? "স্থানীয় প্রোটোটাইপ সিমুলেশন · Python API অনুপলব্ধ";
  const latestPrice = history.at(-1)?.price ?? 0;
  const priceChange = latestPrice - (history.at(-2)?.price ?? latestPrice);
  const chartData = useMemo(() => [
    ...history.map((item) => ({ ...item, observed: item.price })),
    ...forecast.map((item) => ({ ...item, projected: item.price })),
  ], [history, forecast]);
  const guide = cropGuides[crop] ?? cropGuides["ধান"];

  useEffect(() => {
    const controller = new AbortController();
    async function getWeather() {
      setWeatherLoading(true);
      setWeatherError("");
      try {
        const response = await fetch(`/api/weather?lat=${coordinates.lat.toFixed(4)}&lng=${coordinates.lng.toFixed(4)}`, { signal: controller.signal });
        const data = await response.json() as WeatherData & { error?: string };
        if (!response.ok) throw new Error(data.error ?? "আবহাওয়ার তথ্য পাওয়া যায়নি।");
        setWeather(data);
      } catch (error) {
        if ((error as Error).name !== "AbortError") setWeatherError(error instanceof Error ? error.message : "আবহাওয়ার তথ্য পাওয়া যায়নি।");
      } finally {
        if (!controller.signal.aborted) setWeatherLoading(false);
      }
    }
    void getWeather();
    return () => controller.abort();
  }, [coordinates]);

  useEffect(() => {
    const controller = new AbortController();
    setPriceData(null);
    setPriceLoading(true);
    async function getPriceForecast() {
      const query = new URLSearchParams({ district: districtName, crop });
      try {
        const response = await fetch(`/api/forecast?${query.toString()}`, { signal: controller.signal, cache: "no-store" });
        const data = await response.json() as PriceApiResponse & { error?: string };
        if (!response.ok || !Array.isArray(data.history) || !Array.isArray(data.forecast) || !data.source) {
          throw new Error(data.error ?? "নমুনা পূর্বাভাস পাওয়া যায়নি।");
        }
        setPriceData(data);
      } catch {
        if (!controller.signal.aborted) setPriceData(null);
      } finally {
        if (!controller.signal.aborted) setPriceLoading(false);
      }
    }
    void getPriceForecast();
    return () => controller.abort();
  }, [districtName, crop]);

  useEffect(() => {
    return () => {
      if (plantImage) URL.revokeObjectURL(plantImage);
    };
  }, [plantImage]);

  useEffect(() => () => diseaseAbort.current?.abort(), []);

  function selectDistrict(name: string) {
    const district = districts.find((item) => item.name === name) ?? districts[0];
    setDistrictName(district.name);
    setCoordinates({ lat: district.lat, lng: district.lng, source: "district" });
    setLocationMessage(`${district.name} জেলার অবস্থান অনুযায়ী পূর্বাভাস দেখা হচ্ছে`);
  }

  function detectLocation() {
    if (!navigator.geolocation) {
      setLocationMessage("এই ব্রাউজারে অবস্থান সেবা নেই। তালিকা থেকে জেলা বেছে নিন।");
      return;
    }
    setLocating(true);
    setLocationMessage("আপনার ডিভাইস থেকে অবস্থান নেওয়া হচ্ছে…");
    navigator.geolocation.getCurrentPosition(
      ({ coords }) => {
        if (coords.latitude < 20.3 || coords.latitude > 26.9 || coords.longitude < 88 || coords.longitude > 93.7) {
          setLocationMessage("অবস্থানটি বাংলাদেশের সীমানার বাইরে। তালিকা থেকে বাংলাদেশের জেলা বেছে নিন।");
          setLocating(false);
          return;
        }
        const nearby = nearestDistrict(coords.latitude, coords.longitude);
        setCoordinates({ lat: coords.latitude, lng: coords.longitude, source: "device" });
        setDistrictName(nearby.name);
        setLocationMessage(`আপনার অবস্থানের নিকটতম জেলা: ${nearby.name}। মানচিত্রে সঠিক অবস্থান দেখানো হয়েছে।`);
        setLocating(false);
      },
      () => {
        setLocationMessage("অবস্থান অনুমতি পাওয়া যায়নি। ব্রাউজার সেটিংস থেকে অনুমতি দিন বা জেলা বেছে নিন।");
        setLocating(false);
      },
      { enableHighAccuracy: true, timeout: 12000, maximumAge: 300000 },
    );
  }

  async function requestAssistant(messages: ChatMessage[], context: string) {
    const response = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages, context }),
    });
    const data = await response.json() as { answer?: string; error?: string };
    if (!response.ok || !data.answer) throw new Error(data.error ?? "AI সেবাটি সাড়া দিচ্ছে না।");
    return data.answer;
  }

  async function sendChat(event?: FormEvent) {
    event?.preventDefault();
    const question = chatInput.trim();
    if (!question || chatLoading) return;
    const next = [...chatMessages, { role: "user" as const, content: question }];
    setChatMessages(next);
    setChatInput("");
    setChatLoading(true);
    setChatError("");
    try {
      const answer = await requestAssistant(next, `নির্বাচিত জেলা: ${districtName}; ফসল: ${crop}. বাজারদরের অংশটি নমুনা/প্রোটোটাইপ ডেটা, লাইভ দাম নয়।`);
      setChatMessages((current) => [...current, { role: "assistant", content: answer }]);
    } catch (error) {
      setChatError(error instanceof Error ? error.message : "AI সেবাটি সাড়া দিচ্ছে না।");
    } finally {
      setChatLoading(false);
    }
  }

  async function analyzePlantFile(file: File) {
    const supportedImage = file.type.startsWith("image/") && file.type !== "image/svg+xml";
    if (!supportedImage || file.size > 8 * 1024 * 1024) {
      setDiseaseError("৮ এমবি-র মধ্যে JPG, PNG, WebP বা ব্রাউজারে পড়া যায় এমন ছবি দিন।");
      return;
    }

    diseaseAbort.current?.abort();
    const requestId = diseaseRequestId.current + 1;
    diseaseRequestId.current = requestId;
    const controller = new AbortController();
    diseaseAbort.current = controller;
    setPlantFile(file);
    setPlantImage(URL.createObjectURL(file));
    setPlantFileName(file.name || "নির্বাচিত গাছের ছবি");
    setDiseaseResult(null);
    setDiseaseError("");
    setDiseaseLoading(true);

    try {
      const imageData = await fileToJpegDataUrl(file);
      const response = await fetch("/api/disease", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_data_url: imageData,
          symptom,
          include_ai_explanation: includeAiExplanation,
        }),
        signal: controller.signal,
      });
      const data = await response.json() as DiseaseResponse & { error?: string };
      if (!response.ok || !data.prediction || !data.diagnosis || !data.guidance || !data.vision_analysis) {
        throw new Error(data.error ?? "ছবি বিশ্লেষণ করা যায়নি।");
      }
      if (requestId === diseaseRequestId.current) setDiseaseResult(data);
    } catch (error) {
      if ((error as Error).name !== "AbortError" && requestId === diseaseRequestId.current) {
        setDiseaseError(error instanceof Error ? error.message : "এখন ছবি বিশ্লেষণ করা যাচ্ছে না।");
      }
    } finally {
      if (requestId === diseaseRequestId.current) {
        setDiseaseLoading(false);
        diseaseAbort.current = null;
      }
    }
  }

  function onPlantImage(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) void analyzePlantFile(file);
  }

  function onPlantDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    setIsDragOver(false);
    const file = event.dataTransfer.files?.[0];
    if (file) void analyzePlantFile(file);
  }

  function clearPlantImage() {
    diseaseAbort.current?.abort();
    diseaseRequestId.current += 1;
    setDiseaseLoading(false);
    setPlantImage(null);
    setPlantFile(null);
    setPlantFileName("");
    setDiseaseResult(null);
    setDiseaseError("");
  }

  function reanalysePlant() {
    if (plantFile) void analyzePlantFile(plantFile);
  }

  const currentWeather = weather?.weather.current;
  const currentWeatherInfo = wmo(currentWeather?.weather_code ?? 2);
  const WeatherIcon = currentWeatherInfo.icon;

  return (
    <main className="app-shell">
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="brand-lockup">
          <div className="brand-mark"><NextImage src="/brand-logo.png" alt="স্মার্ট কৃষি বুদ্ধিমত্তা" width={52} height={52} priority /></div>
          <div className="brand-copy"><strong>স্মার্ট কৃষি বুদ্ধিমত্তা</strong><span>BNMPC IT Club</span></div>
          <button className="icon-button close-menu" onClick={() => setMobileNav(false)} aria-label="মেনু বন্ধ করুন"><X size={19} /></button>
        </div>
        <div className="sidebar-label">মূল নেভিগেশন</div>
        <nav className="nav-list" aria-label="প্রধান মেনু">
          {navigation.map((item) => {
            const Icon = item.icon;
            return <button key={item.id} onClick={() => { setActive(item.id); setMobileNav(false); }} className={`nav-item ${active === item.id ? "active" : ""}`}>
              <Icon size={19} strokeWidth={active === item.id ? 2.5 : 2} /><span>{item.label}</span>{active === item.id && <ChevronRight size={16} className="nav-arrow" />}
            </button>;
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="side-status"><span className="status-dot" /> সবার জন্য সহজ কৃষি সহায়তা</div>
          <p>আবহাওয়া: Open-Meteo<br />মানচিত্র: OpenStreetMap</p>
        </div>
      </aside>
      {mobileNav && <button aria-label="মেনু বন্ধ করুন" className="mobile-backdrop" onClick={() => setMobileNav(false)} />}

      <section className="content-area">
        <header className="topbar">
          <button className="icon-button hamburger" onClick={() => setMobileNav(true)} aria-label="মেনু খুলুন"><Menu size={22} /></button>
          <div className="breadcrumb"><span>স্মার্ট কৃষি বুদ্ধিমত্তা</span><ChevronRight size={14} /><strong>{navigation.find((item) => item.id === active)?.label}</strong></div>
          <div className="topbar-actions"><div className="live-label"><span className="status-dot" /> সেবা চালু</div><button className="profile-button" onClick={() => setActive("chat")}><Bot size={17} /><span>Krittim-S1</span></button></div>
        </header>

        <div className="page-content">
          {active === "dashboard" && <Dashboard
            districtName={districtName} crop={crop} setActive={setActive} selectDistrict={selectDistrict}
            weather={weather} weatherLoading={weatherLoading} weatherError={weatherError} currentInfo={currentWeatherInfo}
            latestPrice={latestPrice} priceChange={priceChange} history={history}
          />}
          {active === "market" && <MarketPrice districtName={districtName} crop={crop} selectDistrict={selectDistrict} setCrop={setCrop} history={history} />}
          {active === "forecast" && <Forecast districtName={districtName} crop={crop} selectDistrict={selectDistrict} setCrop={setCrop} latestPrice={latestPrice} forecast={forecast} chartData={chartData} forecastSource={forecastSource} priceLoading={priceLoading} />}
          {active === "weather" && <Weather
            districtName={districtName} coordinates={coordinates} weather={weather} loading={weatherLoading} error={weatherError}
            selectDistrict={selectDistrict} detectLocation={detectLocation} locating={locating} locationMessage={locationMessage}
          />}
          {active === "disease" && <DiseaseAssistant
            plantImage={plantImage} plantFileName={plantFileName} galleryInput={galleryInput} cameraInput={cameraInput}
            onImage={onPlantImage} onDrop={onPlantDrop} isDragOver={isDragOver} setIsDragOver={setIsDragOver}
            symptom={symptom} setSymptom={setSymptom} includeAiExplanation={includeAiExplanation}
            setIncludeAiExplanation={setIncludeAiExplanation} result={diseaseResult} error={diseaseError}
            loading={diseaseLoading} clearImage={clearPlantImage} reanalyse={reanalysePlant}
          />}
          {active === "chat" && <Chat
            messages={chatMessages} input={chatInput} setInput={setChatInput} onSubmit={sendChat} loading={chatLoading} error={chatError}
            districtName={districtName} setCrop={setCrop} setActive={setActive}
          />}
          {active === "advice" && <CropAdvice crop={crop} districtName={districtName} setCrop={setCrop} selectDistrict={selectDistrict} guide={guide} />}
        </div>
      </section>
    </main>
  );
}

function ScreenTitle({ eyebrow, title, children }: { eyebrow: string; title: string; children?: React.ReactNode }) {
  return <div className="screen-title"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1></div>{children}</div>;
}

function DistrictSelect({ value, onChange, label = "জেলা নির্বাচন" }: { value: string; onChange: (value: string) => void; label?: string }) {
  const grouped = districts.reduce<Record<string, typeof districts>>((groups, district) => { (groups[district.division] ??= []).push(district); return groups; }, {});
  return <label className="field-control"><span>{label}</span><div className="select-wrap"><MapPin size={17} /><select value={value} onChange={(event) => onChange(event.target.value)}>{Object.entries(grouped).map(([division, entries]) => <optgroup label={`${division} বিভাগ`} key={division}>{entries.map((entry) => <option value={entry.name} key={entry.name}>{entry.name}</option>)}</optgroup>)}</select></div></label>;
}

function CropSelect({ value, onChange, label = "ফসল নির্বাচন" }: { value: string; onChange: (value: string) => void; label?: string }) {
  return <label className="field-control"><span>{label}</span><div className="select-wrap"><Leaf size={17} /><select value={value} onChange={(event) => onChange(event.target.value)}>{crops.map((entry) => <option value={entry} key={entry}>{entry}</option>)}</select></div></label>;
}

function Dashboard({ districtName, crop, setActive, selectDistrict, weather, weatherLoading, weatherError, currentInfo, latestPrice, priceChange, history }: {
  districtName: string; crop: string; setActive: (screen: Screen) => void; selectDistrict: (value: string) => void; weather: WeatherData | null; weatherLoading: boolean; weatherError: string; currentInfo: ReturnType<typeof wmo>; latestPrice: number; priceChange: number; history: Array<{ day: string; price: number }>;
}) {
  const CurrentIcon = currentInfo.icon;
  return <>
    <ScreenTitle eyebrow="বাংলাদেশের কৃষকের জন্য ডিজিটাল সহায়তা" title="আপনার ক্ষেত, আপনার সিদ্ধান্ত"><DistrictSelect value={districtName} onChange={selectDistrict} /></ScreenTitle>
    <section className="hero-panel">
      <div className="hero-copy"><div className="hero-kicker"><Sparkles size={16} /> কৃষি পরিকল্পনার স্মার্ট শুরু</div><h2>আজকের তথ্য দিয়ে<br /><em>আগামীর ফসল</em> গুছিয়ে নিন।</h2><p>আবহাওয়া, বাজারের নমুনা প্রবণতা ও ফসল পরিচর্যার প্রয়োজনীয় তথ্য একই জায়গায়—সহজ বাংলায়।</p><div className="hero-actions"><button className="primary-button" onClick={() => setActive("weather")}>আবহাওয়া দেখুন <ChevronRight size={17} /></button><button className="text-button" onClick={() => setActive("chat")}>AI-কে প্রশ্ন করুন</button></div></div>
      <div className="hero-orb"><div className="hero-leaf leaf-one" /><div className="hero-leaf leaf-two" /><div className="hero-sun" /><div className="hero-field"><span /><span /><span /></div><div className="hero-location"><MapPin size={17} /><div><small>নির্বাচিত এলাকা</small><strong>{districtName}</strong></div></div></div>
    </section>
    <section className="metrics-grid">
      <article className="metric-card weather-metric"><div className="metric-icon sun"><CurrentIcon size={22} /></div><div><p>বর্তমান আবহাওয়া</p>{weatherLoading ? <span className="loading-line">আপডেট হচ্ছে</span> : weatherError ? <strong className="metric-error">তথ্য অনুপলব্ধ</strong> : <><strong>{formatBnNumber(weather?.weather.current.temperature_2m ?? 0, 1)}°</strong><span>{currentInfo.label} · {districtName}</span></>}</div><button onClick={() => setActive("weather")} aria-label="আবহাওয়া বিস্তারিত"><ChevronRight size={18} /></button></article>
      <article className="metric-card"><div className="metric-icon green"><ShoppingBasket size={21} /></div><div><p>{crop} এর নমুনা মূল্য</p><strong>৳ {formatBnNumber(latestPrice, 1)}</strong><span className={priceChange >= 0 ? "positive" : "negative"}>{priceChange >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />} গত দিনের তুলনায় {formatBnNumber(Math.abs(priceChange), 1)} টাকা</span></div><button onClick={() => setActive("market")} aria-label="বাজারদর বিস্তারিত"><ChevronRight size={18} /></button></article>
      <article className="metric-card"><div className="metric-icon blue"><Sprout size={21} /></div><div><p>ফসল পরিচর্যা</p><strong>আজকের করণীয়</strong><span>আবহাওয়া দেখে সেচ পরিকল্পনা করুন</span></div><button onClick={() => setActive("advice")} aria-label="ফসল পরামর্শ বিস্তারিত"><ChevronRight size={18} /></button></article>
    </section>
    <section className="dashboard-grid">
      <article className="panel trend-panel"><div className="panel-heading"><div><p className="section-kicker">নমুনা বাজার পর্যবেক্ষণ</p><h2>{crop} এর মূল্য প্রবণতা</h2></div><button className="quiet-link" onClick={() => setActive("forecast")}>পূর্বাভাস <MoveUpRight size={15} /></button></div><div className="mini-chart"><ResponsiveContainer width="100%" height={220}><AreaChart data={history}><defs><linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#218a63" stopOpacity={0.27}/><stop offset="100%" stopColor="#218a63" stopOpacity={0}/></linearGradient></defs><XAxis dataKey="day" tickLine={false} axisLine={false} tick={{ fill: "#728177", fontSize: 11 }} interval={2}/><YAxis hide domain={["dataMin - 4", "dataMax + 4"]}/><Tooltip formatter={(value) => [`৳ ${formatBnNumber(Number(value), 1)}`, "নমুনা মূল্য"]} labelStyle={{ color: "#17362a" }} contentStyle={{ borderRadius: 12, border: "1px solid #e1e9e4" }}/><Area type="monotone" dataKey="price" stroke="#218a63" strokeWidth={2.5} fill="url(#priceGradient)" /></AreaChart></ResponsiveContainer></div><p className="chart-caption"><FileWarning size={14} /> প্রদর্শিত মূল্য প্রোটোটাইপ ডেটা; এটি লাইভ বাজারদর নয়।</p></article>
      <article className="panel shortcuts-panel"><div className="panel-heading"><div><p className="section-kicker">দ্রুত কাজ</p><h2>যা করতে চান</h2></div></div><div className="shortcut-grid"><button onClick={() => setActive("market")}><ShoppingBasket size={20}/><span>বাজারদর</span><ChevronRight size={15}/></button><button onClick={() => setActive("forecast")}><TrendingUp size={20}/><span>মূল্য পূর্বাভাস</span><ChevronRight size={15}/></button><button onClick={() => setActive("disease")}><ScanLine size={20}/><span>রোগ সহায়তা</span><ChevronRight size={15}/></button><button onClick={() => setActive("chat")}><MessageCircle size={20}/><span>AI সহায়ক</span><ChevronRight size={15}/></button></div></article>
    </section>
  </>;
}

function MarketPrice({ districtName, crop, selectDistrict, setCrop, history }: { districtName: string; crop: string; selectDistrict: (value: string) => void; setCrop: (value: string) => void; history: Array<{ day: string; price: number }> }) {
  const displayed = [crop, "আলু", "পেঁয়াজ", "টমেটো", "বেগুন"].filter((value, index, all) => all.indexOf(value) === index);
  return <>
    <ScreenTitle eyebrow="প্রোটোটাইপ বাজার তথ্য" title="আজকের বাজারদর"><div className="screen-filters"><DistrictSelect value={districtName} onChange={selectDistrict} /><CropSelect value={crop} onChange={setCrop} /></div></ScreenTitle>
    <div className="notice info-notice"><FileWarning size={19}/><div><strong>স্বচ্ছতা নোট</strong><p>এখানের দামগুলো ডেমো/প্রোটোটাইপ ডেটা, লাইভ বা সরকারি বাজারদর নয়। কেনাবেচার আগে স্থানীয় বাজার ও নির্ভরযোগ্য উৎসে যাচাই করুন।</p></div></div>
    <section className="market-overview"><div><p className="section-kicker">নির্বাচিত বাজার এলাকা</p><h2>{districtName} জেলা</h2><span>প্রতি কেজি · সিমুলেটেড নমুনা · লাইভ নয়</span></div><div className="market-stat"><span>নির্বাচিত পণ্য</span><strong>{crop}</strong></div><div className="market-stat"><span>সর্বশেষ নমুনা মূল্য</span><strong>৳ {formatBnNumber(history.at(-1)?.price ?? 0, 1)}</strong></div></section>
    <section className="market-table panel"><div className="table-head"><div><h2>প্রয়োজনীয় পণ্যের তালিকা</h2><p>একই পণ্যের পুনরাবৃত্তি ছাড়া একটি সংক্ষিপ্ত তুলনা</p></div><span className="data-badge">নমুনা ডেটা</span></div><div className="table-scroll"><table><thead><tr><th>পণ্য</th><th>নমুনা মূল্য</th><th>একক</th><th>প্রবণতা</th><th>তথ্যের ধরন</th></tr></thead><tbody>{displayed.map((item) => { const itemHistory = item === crop ? history : createPriceHistory(item, districtName); const price = itemHistory.at(-1)?.price ?? 0; const delta = price - (itemHistory.at(-2)?.price ?? price); return <tr key={item}><td><span className="commodity-icon"><Leaf size={15}/></span>{item}</td><td className="price-cell">৳ {formatBnNumber(price, 1)}</td><td>প্রতি কেজি</td><td><span className={delta >= 0 ? "trend-up" : "trend-down"}>{delta >= 0 ? <TrendingUp size={14}/> : <TrendingDown size={14}/>} {formatBnNumber(Math.abs(delta), 1)}</span></td><td><span className="sample-tag">প্রোটোটাইপ</span></td></tr>; })}</tbody></table></div></section>
    <div className="source-strip"><ShieldCheck size={19}/><div><strong>লাইভ ডেটা সংযোগের পরিকল্পনা</strong><span>সরকারি/বিশ্বস্ত উৎস নির্ভরযোগ্যভাবে পাওয়া গেলে উৎস, রিপোর্টের তারিখ এবং হালনাগাদের সময় আলাদাভাবে দেখানো হবে।</span></div></div>
  </>;
}

function Forecast({ districtName, crop, selectDistrict, setCrop, latestPrice, forecast, chartData, forecastSource, priceLoading }: { districtName: string; crop: string; selectDistrict: (value: string) => void; setCrop: (value: string) => void; latestPrice: number; forecast: Array<{ day: string; price: number }>; chartData: Array<{ day: string; observed?: number; projected?: number }>; forecastSource: string; priceLoading: boolean }) {
  const predictedAverage = forecast.reduce((sum, item) => sum + item.price, 0) / forecast.length;
  const delta = predictedAverage - latestPrice;
  return <>
    <ScreenTitle eyebrow="পরীক্ষামূলক মূল্য বিশ্লেষণ" title="মূল্য পূর্বাভাস"><div className="screen-filters"><DistrictSelect value={districtName} onChange={selectDistrict}/><CropSelect value={crop} onChange={setCrop}/></div></ScreenTitle>
    <div className="notice warning-notice"><Gauge size={19}/><div><strong>পূর্বাভাস, প্রতিশ্রুতি নয়</strong><p>এটি সংযুক্ত নমুনা CSV বা ডেমো ভিত্তিতে তৈরি সিমুলেশন—লাইভ বা সরকারি বাজারদর নয়। CSV-র রেকর্ড ২০২৫ সালের মে মাসের পরে নেই; চার্টের তারিখ ও ১৪ দিনের রেখা বর্তমান বাজারের পর্যবেক্ষণ বা যাচাইকৃত পূর্বাভাস নয়। কেনাবেচার আগে স্থানীয় বাজার ও কৃষি কর্মকর্তার তথ্য যাচাই করুন।</p></div></div>
    <section className="forecast-stats"><article><span>সাম্প্রতিক নমুনা মূল্য</span><strong>৳ {formatBnNumber(latestPrice, 1)}</strong><small>{priceLoading ? "সার্ভার থেকে নমুনা আনা হচ্ছে…" : forecastSource}</small></article><article><span>১৪ দিনের গড় পূর্বাভাস</span><strong>৳ {formatBnNumber(predictedAverage, 1)}</strong><small>প্রতি কেজি</small></article><article><span>সম্ভাব্য পরিবর্তন</span><strong className={delta >= 0 ? "positive-text" : "negative-text"}>{delta >= 0 ? "+" : ""}{formatBnNumber(delta, 1)} টাকা</strong><small>সাম্প্রতিক নমুনা দামের তুলনায়</small></article></section>
    <section className="panel forecast-chart"><div className="panel-heading"><div><p className="section-kicker">{districtName} · {crop}</p><h2>পর্যবেক্ষণ ও পূর্বাভাস</h2></div><div className="legend"><span><i className="observed-dot"/>CSV/ডেমো সিমুলেশন</span><span><i className="forecast-dot"/>পরীক্ষামূলক পূর্বাভাস</span></div></div><ResponsiveContainer width="100%" height={330}><LineChart data={chartData} margin={{ top: 10, right: 6, left: -20, bottom: 0 }}><CartesianGrid stroke="#edf1ed" vertical={false}/><XAxis dataKey="day" tickLine={false} axisLine={false} tick={{ fill: "#728177", fontSize: 11 }} interval={2}/><YAxis tickLine={false} axisLine={false} tick={{ fill: "#728177", fontSize: 11 }}/><Tooltip formatter={(value) => [`৳ ${formatBnNumber(Number(value), 1)}`, ""]} contentStyle={{ borderRadius: 12, border: "1px solid #e1e9e4" }}/><Line type="monotone" dataKey="observed" name="পর্যবেক্ষণ" stroke="#218a63" strokeWidth={3} dot={false} activeDot={{ r: 5 }}/><Line type="monotone" dataKey="projected" name="পূর্বাভাস" stroke="#d68b2a" strokeWidth={3} strokeDasharray="7 5" dot={false} activeDot={{ r: 5 }}/></LineChart></ResponsiveContainer></section>
    <section className="forecast-days">{forecast.slice(0, 7).map((item, index) => <div key={item.day} className={index === 0 ? "today" : ""}><span>{index === 0 ? "আগামীকাল" : item.day}</span><strong>৳ {formatBnNumber(item.price, 1)}</strong><small>প্রতি কেজি</small></div>)}</section>
  </>;
}

function Weather({ districtName, coordinates, weather, loading, error, selectDistrict, detectLocation, locating, locationMessage }: { districtName: string; coordinates: Coordinates; weather: WeatherData | null; loading: boolean; error: string; selectDistrict: (value: string) => void; detectLocation: () => void; locating: boolean; locationMessage: string }) {
  const current = weather?.weather.current;
  const weatherInfo = wmo(current?.weather_code ?? 2);
  const MainWeatherIcon = weatherInfo.icon;
  const rainfallDays = weather?.weather.daily.precipitation_probability_max.filter((value) => value >= 50).length ?? 0;
  return <>
    <ScreenTitle eyebrow="Open-Meteo থেকে পূর্বাভাস" title="আবহাওয়া ও অবস্থান"><DistrictSelect value={districtName} onChange={selectDistrict}/></ScreenTitle>
    <section className="location-panel panel"><div><p className="section-kicker">সঠিক জায়গা, প্রাসঙ্গিক পূর্বাভাস</p><h2>আপনি কোথায় আছেন?</h2><p>{locationMessage}</p><button className="primary-button" onClick={detectLocation} disabled={locating}>{locating ? <LoaderCircle className="spin" size={17}/> : <Crosshair size={17}/>} {locating ? "অবস্থান নেওয়া হচ্ছে" : "আমার অবস্থান ব্যবহার করুন"}</button><small><MapPin size={13}/> অনুমতি দিলে আপনার ডিভাইসের অবস্থান ব্যবহার করা হবে; কোনো API key প্রয়োজন নেই।</small></div><div className="map-frame"><iframe title="OpenStreetMap অবস্থান" src={mapUrl(coordinates.lat, coordinates.lng)} loading="lazy"/><a href={`https://www.openstreetmap.org/?mlat=${coordinates.lat}&mlon=${coordinates.lng}#map=10/${coordinates.lat}/${coordinates.lng}`} target="_blank" rel="noreferrer"><Map size={15}/> OpenStreetMap-এ বড় করে দেখুন</a></div></section>
    {loading ? <div className="weather-loading panel"><LoaderCircle className="spin" size={21}/> সর্বশেষ আবহাওয়ার তথ্য আনা হচ্ছে…</div> : error ? <div className="notice warning-notice"><FileWarning size={19}/><div><strong>আবহাওয়া সাময়িকভাবে অনুপলব্ধ</strong><p>{error}</p></div></div> : current && weather ? <>
      <section className="weather-now"><article className="weather-main"><div className={`weather-symbol ${weatherInfo.color}`}><MainWeatherIcon size={49}/></div><div><p>{districtName} · {coordinates.source === "device" ? "আপনার অবস্থান" : "জেলা কেন্দ্র"}</p><strong>{formatBnNumber(current.temperature_2m, 1)}°</strong><span>অনুভূত হচ্ছে {formatBnNumber(current.apparent_temperature, 1)}° · {weatherInfo.label}</span></div></article><article><ThermometerSun size={21}/><span>আর্দ্রতা</span><strong>{formatBnNumber(current.relative_humidity_2m)}%</strong></article><article><Wind size={21}/><span>বাতাসের গতি</span><strong>{formatBnNumber(current.wind_speed_10m, 1)} কিমি/ঘ.</strong></article><article><Droplets size={21}/><span>বৃষ্টিপাত</span><strong>{formatBnNumber(current.precipitation, 1)} মিমি</strong></article></section>
      <section className="weather-forecast panel"><div className="panel-heading"><div><p className="section-kicker">সাত দিনের পরিকল্পনা</p><h2>সহজ পূর্বাভাস</h2></div><span className="source-label">উৎস: {weather.source}</span></div><div className="forecast-grid">{weather.weather.daily.time.map((date, index) => { const info = wmo(weather.weather.daily.weather_code[index]); const Icon = info.icon; return <article key={date} className={index === 0 ? "forecast-today" : ""}><p>{index === 0 ? "আজ" : banglaDate(date, true)}</p><Icon className={`weather-small ${info.color}`} size={25}/><strong>{formatBnNumber(weather.weather.daily.temperature_2m_max[index])}° <small>{formatBnNumber(weather.weather.daily.temperature_2m_min[index])}°</small></strong><span><Droplets size={13}/>{formatBnNumber(weather.weather.daily.precipitation_probability_max[index])}%</span><em>{info.label}</em></article>; })}</div></section>
      <section className="weather-advisory"><div className="advisory-icon"><Sprout size={22}/></div><div><p className="section-kicker">খেতের জন্য সতর্ক বার্তা</p><h2>{rainfallDays > 0 ? `আগামী ৭ দিনে ${formatBnNumber(rainfallDays)} দিন বৃষ্টির সম্ভাবনা আছে` : "আগামী ৭ দিনে বড় বৃষ্টির সম্ভাবনা কম"}</h2><p>{rainfallDays > 0 ? "নালা ও পানি নিষ্কাশনের পথ পরিষ্কার রাখুন। স্প্রে বা সার প্রয়োগের আগে স্থানীয় আবহাওয়া পরিস্থিতি ও পণ্যের লেবেল বিবেচনা করুন।" : "মাটির আর্দ্রতা পরীক্ষা করে প্রয়োজনমতো সেচের সময় ঠিক করুন।"}</p></div></section>
    </> : null}
  </>;
}

function MarkdownContent({ content, className = "" }: { content: string; className?: string }) {
  return <div className={`markdown-content ${className}`}>
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
      }}
    >{content}</ReactMarkdown>
  </div>;
}

function DiseaseAssistant({
  plantImage, plantFileName, galleryInput, cameraInput, onImage, onDrop, isDragOver, setIsDragOver,
  symptom, setSymptom, includeAiExplanation, setIncludeAiExplanation, result, error, loading, clearImage, reanalyse,
}: {
  plantImage: string | null;
  plantFileName: string;
  galleryInput: React.RefObject<HTMLInputElement | null>;
  cameraInput: React.RefObject<HTMLInputElement | null>;
  onImage: (event: ChangeEvent<HTMLInputElement>) => void;
  onDrop: (event: DragEvent<HTMLElement>) => void;
  isDragOver: boolean;
  setIsDragOver: (value: boolean) => void;
  symptom: string;
  setSymptom: (value: string) => void;
  includeAiExplanation: boolean;
  setIncludeAiExplanation: (value: boolean) => void;
  result: DiseaseResponse | null;
  error: string;
  loading: boolean;
  clearImage: () => void;
  reanalyse: () => void;
}) {
  const diagnosis = result?.diagnosis;
  const healthy = (diagnosis?.leaf_status === "leaf" && diagnosis.cause_type === "healthy") || result?.guidance.category === "healthy_class";
  const actions = diagnosis?.nonchemical_actions_bn.length
    ? diagnosis.nonchemical_actions_bn.slice(0, 2)
    : (result?.guidance.actions ?? []).slice(0, 2);
  const onDragLeave = (event: DragEvent<HTMLDivElement>) => {
    const nextTarget = event.relatedTarget;
    if (!(nextTarget instanceof Node) || !event.currentTarget.contains(nextTarget)) setIsDragOver(false);
  };

  return <>
    <ScreenTitle eyebrow="ছবি দিন, প্রাথমিক সহায়তা পান" title="রোগ সহায়তা" />
    <div className="notice info-notice disease-scope-notice"><ShieldCheck size={19}/><div><strong>Krittim AI Vision · ছবিভিত্তিক প্রাথমিক যাচাই</strong><p>বিশ্লেষণ চালু থাকলে আপলোড করা ছবি ও আপনার লেখা লক্ষণ সংযুক্ত দূরবর্তী AI সেবায় পাঠানো হতে পারে। স্থানীয় ResNet9-এর ৩৮টি PlantVillage শ্রেণি সীমিত সহায়ক—বাংলাদেশের সব ফসল বা রোগ এতে নেই। এক ছবিতে নিশ্চিত রোগনির্ণয় হয় না; ব্যক্তিগত বা পরিচয়-দেখা যায় এমন ছবি দেবেন না।</p></div></div>
    <div className="diagnosis-flow" aria-label="রোগ সহায়তা ব্যবহারের ধাপ">
      <div><span>১</span><strong>পাতার ছবি দিন</strong></div><ChevronRight size={18} aria-hidden="true" />
      <div><span>২</span><strong>লক্ষণ লিখুন <em>(ঐচ্ছিক)</em></strong></div><ChevronRight size={18} aria-hidden="true" />
      <div><span>৩</span><strong>ফল মিলিয়ে দেখুন</strong></div>
    </div>

    <section className="disease-layout">
      <div className="upload-card panel">
        <div className="upload-heading"><div className="metric-icon green"><ScanLine size={22}/></div><div><h2>পাতার ছবি যোগ করুন</h2><p>ভালো আলোতে আক্রান্ত অংশটি কাছে থেকে তুলুন</p></div></div>
        <input ref={galleryInput} type="file" accept="image/*" onChange={onImage} hidden />
        <input ref={cameraInput} type="file" accept="image/*" capture="environment" onChange={onImage} hidden />
        <div
          className={`disease-drop-area ${isDragOver ? "drag-over" : ""}`}
          onDragEnter={(event) => { event.preventDefault(); setIsDragOver(true); }}
          onDragOver={(event) => { event.preventDefault(); setIsDragOver(true); }}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
        >
          {plantImage ? <div className="image-preview">
            <img src={plantImage} alt="আপলোড করা গাছের পাতার প্রিভিউ" />
            <div className="preview-actions"><span title={plantFileName}>{plantFileName}</span><button type="button" onClick={clearImage} aria-label="ছবি সরান"><X size={15}/></button></div>
          </div> : <div className="upload-zone">
            <span className="upload-icon"><Upload size={25}/></span>
            <strong>ছবি এখানে টেনে আনুন</strong>
            <small>অথবা গ্যালারি থেকে বাছুন · JPG, PNG, WebP · সর্বোচ্চ ৮ এমবি</small>
          </div>}
        </div>
        <div className="capture-actions">
          <button type="button" className="secondary-capture" onClick={() => galleryInput.current?.click()}><ImagePlus size={17}/> গ্যালারি</button>
          <button type="button" className="secondary-capture" onClick={() => cameraInput.current?.click()}><Camera size={17}/> ক্যামেরা</button>
          {plantImage && <button type="button" className="secondary-capture" onClick={() => galleryInput.current?.click()}><Upload size={16}/> ছবি বদলান</button>}
        </div>
        {loading && <p className="analysis-status"><LoaderCircle size={16} className="spin"/> ছবিটি বিশ্লেষণ করা হচ্ছে…</p>}
        {error && <div className="disease-error" role="alert"><FileWarning size={16}/><span>{error}</span></div>}
      </div>

      <div className="diagnosis-card panel">
        <div className="panel-heading"><div><p className="section-kicker">ঐচ্ছিক পর্যবেক্ষণ</p><h2>আপনি কী লক্ষ করছেন?</h2></div><Bug size={23} className="muted-icon"/></div>
        <label className="text-field"><span>পাতার দৃশ্যমান লক্ষণ (ঐচ্ছিক)</span><textarea value={symptom} onChange={(event) => setSymptom(event.target.value)} placeholder="যেমন: পাতায় বাদামি দাগ, কুঁকড়ে যাচ্ছে, গাছ দুর্বল দেখাচ্ছে…" rows={5}/></label>
        <label className="ai-option-card"><input type="checkbox" checked={includeAiExplanation} onChange={(event) => setIncludeAiExplanation(event.target.checked)} /><span><strong>Krittim-S1 দিয়ে দীর্ঘ AI ব্যাখ্যা যোগ করুন</strong><small>সংক্ষিপ্ত ভিজ্যুয়াল ফল সবসময় আগে আসে। চালু করলে Krittim AI Vision-এর সারাংশ ও আপনার লেখা লক্ষণ দূরবর্তী text AI সেবায় পাঠানো হবে—ছবিটি নয়; কোনো ওষুধ বা ডোজ দেওয়া হবে না।</small></span><Sparkles size={18}/></label>
        <button type="button" className="primary-button full-width" onClick={reanalyse} disabled={!plantImage || loading}>{loading ? <LoaderCircle size={17} className="spin"/> : <ScanLine size={17}/>} {loading ? "বিশ্লেষণ চলছে" : "লক্ষণ/AI বিকল্পসহ আবার বিশ্লেষণ"}</button>
        <p className="tiny-note"><ShieldCheck size={14}/> ফসল বাছাইয়ের প্রয়োজন নেই। লক্ষণ বা AI বিকল্প বদলালে নতুন বিশ্লেষণ চালান।</p>
      </div>
    </section>

    {result && <section className="disease-results" aria-live="polite">
      <div className="result-heading panel diagnosis-direct-card">
        <div className="result-status-icon"><Leaf size={22}/></div>
        <div className="diagnosis-copy">
          <p className="section-kicker">{diagnosis?.source === "krittim_vision" ? "Krittim AI Vision · প্রাথমিক ছবি-অনুমান" : "স্থানীয় সীমিত মডেল · সহায়ক অনুমান"}</p>
          <h2>{diagnosis?.disease_label || result.prediction.label}</h2>
          <p className="result-cause">{diagnosis?.cause_label_bn || "ছবি থেকে কারণ নির্ধারণ করা যায়নি"}</p>
          <p className="result-score">{result.image_sent_to_vision_service ? result.vision_analysis.available ? "ছবিটি বিশ্লেষণের জন্য দূরবর্তী AI সেবায় পাঠানো হয়েছিল।" : "ছবি AI সেবায় পাঠানোর চেষ্টা হয়েছে, তবে ফল পাওয়া যায়নি।" : "দূরবর্তী ছবি-ভিত্তিক AI চালু নেই; সীমিত স্থানীয় মডেলের ফল দেখানো হচ্ছে।"}</p>
        </div>
      </div>
      {!result.vision_analysis.available && <div className="notice warning-notice result-caveat"><FileWarning size={18}/><div><strong>ছবিভিত্তিক AI ফল পাওয়া যায়নি</strong><p>{result.vision_analysis.status === "not_configured" ? "দূরবর্তী ভিশন AI সেটআপ নেই; সীমিত ৩৮-শ্রেণির স্থানীয় ফল দেখানো হয়েছে।" : result.vision_analysis.status === "invalid_key" ? "ভিশন AI সেবার কনফিগারেশন যাচাই হয়নি; স্থানীয় সীমিত ফল দেখানো হয়েছে।" : result.vision_analysis.status === "model_unavailable" ? "কনফিগার করা ভিশন মডেলটি পাওয়া যায়নি; স্থানীয় সীমিত ফল দেখানো হয়েছে।" : result.vision_analysis.status === "rate_limit" ? "ভিশন AI সেবার অনুরোধসীমা পূর্ণ; পরে আবার চেষ্টা করুন। আপাতত স্থানীয় সীমিত ফল দেখানো হয়েছে।" : "ভিশন AI উত্তর দেয়নি; স্থানীয় ৩৮-শ্রেণির fallback বাংলাদেশের সব ফসল/রোগ শনাক্ত করে না।"}</p></div></div>}
      <div className="notice warning-notice result-caveat"><FileWarning size={19}/><div><strong>এটি নিশ্চিত রোগনির্ণয় নয়</strong><p>{result.scope_warning}</p></div></div>

      <section className="panel result-card quick-prescription-card">
        <div className="panel-heading"><div><p className="section-kicker">ছোট, সরাসরি পরবর্তী পদক্ষেপ</p><h2>{healthy ? "এখনই ওষুধ নয়" : "এখন কী করবেন"}</h2></div><ShieldCheck size={21} className="muted-icon"/></div>
        {diagnosis?.visible_signs_bn?.length ? <div className="signs-check"><strong><span className="detail-index">১</span>ছবিতে AI যে লক্ষণ দেখেছে</strong><ul>{diagnosis.visible_signs_bn.slice(0, 2).map((sign, index) => <li key={index}>{sign}</li>)}</ul></div> : <div className="signs-check"><strong><span className="detail-index">১</span>হাতে মিলিয়ে দেখুন</strong><p>{result.guidance.signs_to_check}</p></div>}
        {diagnosis?.leaf_status === "not_leaf" && <p className="small-caveat">পাতার পরিষ্কার ছবি দিন—এই ফলের ভিত্তিতে কোনো চিকিৎসা করবেন না।</p>}
        <div className="result-grid quick-action-grid">
          <div className="diagnosis-detail-card"><h3><span className="detail-index">২</span>করণীয়</h3><ul>{actions.map((action, index) => <li key={index}>{action}</li>)}</ul></div>
          <div className="diagnosis-detail-card"><h3><span className="detail-index">৩</span>পানি</h3><p>{result.guidance.watering.care}</p><p className="small-caveat">{result.guidance.watering.amount_note}</p></div>
        </div>

        <div className={`quick-medicine ${result.guidance.products.length && !healthy && diagnosis?.leaf_status === "leaf" ? "verified-medicine" : "no-product-match"}`}>
          <h3><span className="detail-index">৪</span>{result.guidance.products.length && !healthy && diagnosis?.leaf_status === "leaf" ? "যাচাইকৃত DAE পণ্য-রেকর্ড" : "ওষুধ / সার"}</h3>
          {result.guidance.products.length && !healthy && diagnosis?.leaf_status === "leaf" ? <>
            {(() => { const product = result.guidance.products[0]; return <>
              <p className="quick-product-name"><strong>{product.name}</strong> · AP-{product.registration_no.replace(/^AP-/, "")}</p>
              <p><strong>শুধু এই ব্যবহার:</strong> {product.label_scope}</p>
              <p><strong>রেকর্ডের মাত্রা:</strong> {product.rate}</p>
              <p><strong>প্রয়োগপথ:</strong> {product.application_method}</p>
              <p className="small-caveat">{result.guidance.product_note}</p>
              <p className="small-caveat">ছবির AI ফল ভুল হতে পারে—ফসল/রোগ বাস্তবে মিললে তবেই বিবেচনা করুন। বর্তমান প্যাকেটের লেবেলই চূড়ান্ত; রেকর্ড মানেই দোকানে মজুত বা নিবন্ধন সক্রিয়—এ নিশ্চয়তা নয়।</p>
              <a className="product-source-link" href={product.source_url} target="_blank" rel="noreferrer">{product.source_label} <MoveUpRight size={14}/></a>
            </>; })()}
          </> : <>
            <p>{healthy ? "ছবিতে স্পষ্ট রোগচিহ্ন নেই; রোগ বা পুষ্টিঘাটতির প্রমাণ ছাড়া ওষুধ/সার দেবেন না।" : "এই ফল থেকে নির্ভরযোগ্য ফসল–রোগ মিল ও DAE-যাচাইকৃত পণ্যের মাত্রা নিশ্চিত করা যায়নি।"}</p>
            <p className="small-caveat">{result.guidance.product_note}</p>
            {!healthy && <p className="small-caveat">নিজে থেকে ব্র্যান্ড, মাত্রা, মিশ্রণ বা পাতায়/গোড়ায় প্রয়োগ ঠিক করবেন না; আগে উপজেলা কৃষি অফিসে যাচাই করুন।</p>}
          </>}
        </div>
      </section>

      <details className="advanced-result-details panel">
        <summary>মডেলের বিকল্প ফল, পণ্যের নির্দেশনা ও উৎস</summary>
        <div className="advanced-result-content">
          <p className="result-explanation">{result.guidance.reason}</p>
          <div className="signs-check"><strong>আরও যে লক্ষণগুলো মিলিয়ে দেখবেন</strong><p>{result.guidance.signs_to_check}</p></div>
          {result.top_predictions.length > 0 ? <div className="top-predictions-card">
            <div className="panel-heading"><div><p className="section-kicker">কেবল স্থানীয় মডেলের বিকল্প</p><h3>শীর্ষ শ্রেণিগুলো</h3></div><span className="model-tag">৩৮-শ্রেণির মডেল</span></div>
            <div className="prediction-list">{result.top_predictions.slice(0, 3).map((item, index) => <div className="prediction-item" key={item.class_label}>
              <span className="prediction-rank">{formatBnNumber(index + 1)}</span><div className="prediction-name"><strong>{item.label}</strong><div className="prediction-track"><i style={{ width: `${Math.max(0, Math.min(100, item.score_percent))}%` }}/></div></div><span className="prediction-score">{formatBnNumber(item.score_percent, 1)}%</span>
            </div>)}</div>
            <p className="small-caveat">এগুলো softmax score, ক্যালিব্রেটেড সম্ভাবনা বা নির্ভুলতা নয়।</p>
          </div> : <p className="small-caveat">স্থানীয় ৩৮-শ্রেণির মডেল এই অনুরোধে চালু ছিল না।</p>}
          {result.guidance.products.map((product) => <article className="product-card" key={product.registration_no}>
            <div className="product-card-heading"><div><h3>{product.name}</h3><span>AP নম্বর: {product.registration_no} · {product.active_ingredients}</span></div><span className="verified-tag">DAE record</span></div>
            <p><strong>লেবেলে উল্লেখিত ব্যবহার:</strong> {product.label_scope}</p><p><strong>রেকর্ডের মাত্রা:</strong> {product.rate}</p><p><strong>প্রয়োগপথ:</strong> {product.application_method}</p>
            <ul>{product.directions.map((direction, index) => <li key={index}>{direction}</li>)}</ul>
            <a className="product-source-link" href={product.source_url} target="_blank" rel="noreferrer">{product.source_label} <MoveUpRight size={14}/></a>
          </article>)}
          <div className="source-links">{result.guidance.sources.map((source, index) => <a key={`${source.url}-${index}`} href={source.url} target="_blank" rel="noreferrer">{source.label}<MoveUpRight size={14}/></a>)}</div>
          <p className="small-caveat">রেফারেন্স শেষ পর্যালোচনা: {result.guidance.source_checked_on} · PRISM-এর তালিকা উপজেলা-ভিত্তিক; এটি বাজারে পণ্য মজুত বা ছবির রোগনির্ণয়ের নিশ্চয়তা নয়।</p>
        </div>
      </details>

      {result.summary_requested && <section className="panel ai-summary-card"><div className="panel-heading"><div><p className="section-kicker">ঐচ্ছিক Krittim-S1 ব্যাখ্যা</p><h2>আরও ব্যাখ্যা</h2></div><Sparkles size={20} className="muted-icon"/></div>{result.summary_available && result.summary ? <MarkdownContent content={result.summary} className="ai-summary-text"/> : <p className="small-caveat">এই মুহূর্তে AI ব্যাখ্যা পাওয়া যায়নি। সংক্ষিপ্ত নিয়মভিত্তিক নির্দেশনা উপরে আছে।</p>}<p className="small-caveat">Krittim-S1 ছবির pixel দেখে না; এটি Krittim AI Vision-এর সংক্ষিপ্ত ফল ও আপনার লেখা লক্ষণ থেকে ব্যাখ্যা দেয়। AI ব্যাখ্যা কোনো প্রেসক্রিপশন নয়।</p></section>}
    </section>}
  </>;
}

function Chat({ messages, input, setInput, onSubmit, loading, error, districtName, setCrop, setActive }: { messages: ChatMessage[]; input: string; setInput: React.Dispatch<React.SetStateAction<string>>; onSubmit: (event: FormEvent) => void; loading: boolean; error: string; districtName: string; setCrop: (value: string) => void; setActive: (value: Screen) => void }) {
  const suggestions = ["ধানের পাতায় দাগ হলে প্রথমে কী দেখব?", "বৃষ্টির আগে কীভাবে সেচ পরিকল্পনা করব?", "আলু সংরক্ষণে কী খেয়াল রাখা উচিত?"];
  const [speechLanguage, setSpeechLanguage] = useState<"bn-BD" | "en-US">("bn-BD");
  const [speechSupported, setSpeechSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [speechMessage, setSpeechMessage] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);

  useEffect(() => {
    const speechWindow = window as SpeechRecognitionWindow;
    const supported = Boolean(speechWindow.SpeechRecognition || speechWindow.webkitSpeechRecognition);
    setSpeechSupported(supported);
    if (!supported) setSpeechMessage("এই ব্রাউজারে speech recognition নেই। Chrome বা Edge-এ চেষ্টা করুন, অথবা লিখে পাঠান।");
    return () => {
      const recognition = recognitionRef.current;
      if (recognition) {
        recognition.onresult = null;
        recognition.onerror = null;
        recognition.onend = null;
        recognition.abort();
      }
      recognitionRef.current = null;
    };
  }, []);

  function chooseSpeechLanguage(language: "bn-BD" | "en-US") {
    if (recognitionRef.current) {
      const recognition = recognitionRef.current;
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.onend = null;
      recognition.abort();
      recognitionRef.current = null;
      setListening(false);
      setInterimTranscript("");
    }
    setSpeechLanguage(language);
    setSpeechMessage(language === "bn-BD" ? "বাংলায় বলুন।" : "Speak in English.");
  }

  function toggleSpeechInput() {
    if (listening) {
      recognitionRef.current?.stop();
      setListening(false);
      setSpeechMessage("কথা লেখা বন্ধ হয়েছে।");
      return;
    }

    const speechWindow = window as SpeechRecognitionWindow;
    const Recognition = speechWindow.SpeechRecognition || speechWindow.webkitSpeechRecognition;
    if (!Recognition) {
      setSpeechSupported(false);
      setSpeechMessage("এই ব্রাউজারে কথা শুনে লেখার সুবিধা নেই। Chrome বা Edge-এ চেষ্টা করুন, অথবা লিখে পাঠান।");
      return;
    }

    try {
      const recognition = new Recognition();
      recognition.lang = speechLanguage;
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.onresult = (event) => {
        let finalText = "";
        let interimText = "";
        for (let index = event.resultIndex; index < event.results.length; index += 1) {
          const transcript = event.results[index][0]?.transcript?.trim() ?? "";
          if (!transcript) continue;
          if (event.results[index].isFinal) finalText += `${finalText ? " " : ""}${transcript}`;
          else interimText += `${interimText ? " " : ""}${transcript}`;
        }
        if (finalText) {
          setInput((current) => `${current}${current && !/\s$/.test(current) ? " " : ""}${finalText}`);
          setSpeechMessage("কথা থেকে লেখা যোগ হয়েছে। চাইলে আরও বলুন।");
        }
        setInterimTranscript(interimText);
      };
      recognition.onerror = (event) => {
        const errorMessages: Record<string, string> = {
          "not-allowed": "মাইক্রোফোন ব্যবহারের অনুমতি দিন, তারপর আবার চেষ্টা করুন।",
          "service-not-allowed": "ব্রাউজারের speech service অনুমতি পায়নি।",
          "audio-capture": "মাইক্রোফোন পাওয়া যায়নি। ডিভাইসের মাইক্রোফোন পরীক্ষা করুন।",
          "no-speech": "কথা শোনা যায়নি। আবার মাইক্রোফোন চাপুন।",
          network: "ভয়েস সেবা সংযোগ পায়নি। পরে চেষ্টা করুন বা লিখে পাঠান।",
          "language-not-supported": "এই ভাষাটি ব্রাউজারে নেই। অন্য ভাষা বেছে নিন।",
        };
        setSpeechMessage(errorMessages[event.error] ?? "কথা শোনা যায়নি। আবার চেষ্টা করুন বা লিখে পাঠান।");
        setListening(false);
        setInterimTranscript("");
        if (recognitionRef.current === recognition) recognitionRef.current = null;
      };
      recognition.onend = () => {
        setListening(false);
        setInterimTranscript("");
        setSpeechMessage((message) => message.startsWith("শুনছি") || message.startsWith("Listening")
          ? "কথা শোনা বন্ধ হয়েছে। আবার শুরু করতে মাইক্রোফোন চাপুন।"
          : message);
        if (recognitionRef.current === recognition) recognitionRef.current = null;
      };
      recognitionRef.current = recognition;
      recognition.start();
      setListening(true);
      setSpeechMessage(speechLanguage === "bn-BD" ? "শুনছি… থামাতে মাইক্রোফোন বোতাম চাপুন।" : "Listening… tap the microphone button to stop.");
    } catch {
      recognitionRef.current = null;
      setListening(false);
      setSpeechMessage("মাইক্রোফোন চালু করা যায়নি। অনুমতি পরীক্ষা করে আবার চেষ্টা করুন।");
    }
  }

  return <>
    <ScreenTitle eyebrow="বাংলায় কৃষির কথা বলুন" title="AI কৃষি সহায়ক"><div className="ai-chip"><Bot size={17}/> Krittim-S1</div></ScreenTitle>
    <section className="chat-layout">
      <aside className="chat-context">
        <div className="assistant-avatar"><Bot size={25}/></div>
        <h2>আজ কী নিয়ে ভাবছেন?</h2>
        <p>ফসলের সাধারণ পরিচর্যা, আবহাওয়া বুঝে পরিকল্পনা এবং লক্ষণভিত্তিক নিরাপদ পরামর্শে আমি সাহায্য করতে পারি।</p>
        <div className="context-item"><MapPin size={17}/><span>এলাকা<strong>{districtName}</strong></span></div>
        <div className="context-item"><ShieldCheck size={17}/><span>উত্তর ভাষা<strong>বাংলা</strong></span></div>
        <button onClick={() => { setCrop("ধান"); setActive("advice"); }} className="quiet-link">ফসলের গাইড দেখুন <ChevronRight size={15}/></button>
      </aside>
      <div className="chat-panel panel">
        <div className="chat-head">
          <div><strong>স্মার্ট কৃষি বুদ্ধিমত্তা</strong><span><i className="status-dot"/> উত্তর দিতে প্রস্তুত</span></div>
          <span>Krittim-S1 · সাধারণ কৃষি সহায়তা</span>
        </div>
        <div className="chat-thread">
          {messages.map((message, index) => <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
            <div className="message-avatar">{message.role === "assistant" ? <Bot size={16}/> : <span>আপনি</span>}</div>
            <div className={`message-bubble ${message.role}`}><MarkdownContent content={message.content}/></div>
          </div>)}
          {loading && <div className="message assistant"><div className="message-avatar"><Bot size={16}/></div><p className="typing"><i/><i/><i/></p></div>}
        </div>
        {error && <div className="chat-error" role="alert"><FileWarning size={16}/><span>{error}</span></div>}
        <div className="suggestion-row">{suggestions.map((suggestion) => <button type="button" key={suggestion} onClick={() => setInput(suggestion)}>{suggestion}</button>)}</div>
        <form className="chat-composer" onSubmit={onSubmit}>
          <div className="voice-toolbar">
            <div className="voice-language-picker" role="group" aria-label="কথার ভাষা বাছুন">
              <span>কথার ভাষা</span>
              <button type="button" className={speechLanguage === "bn-BD" ? "selected" : ""} aria-pressed={speechLanguage === "bn-BD"} onClick={() => chooseSpeechLanguage("bn-BD")}>বাংলা</button>
              <button type="button" className={speechLanguage === "en-US" ? "selected" : ""} aria-pressed={speechLanguage === "en-US"} onClick={() => chooseSpeechLanguage("en-US")}>English</button>
            </div>
            <button type="button" className={`voice-button ${listening ? "listening" : ""}`} onClick={toggleSpeechInput} aria-pressed={listening} aria-label={listening ? "কথা শোনা বন্ধ করুন" : "কথা শুনে লিখুন"} title={speechSupported ? "বাংলা বা ইংরেজিতে বলুন" : "Speech recognition-এর জন্য Chrome বা Edge চেষ্টা করুন"}>
              {listening ? <MicOff size={18}/> : <Mic size={18}/>}<span>{listening ? "থামান" : "বলুন ও লিখুন"}</span>
            </button>
          </div>
          <div className="chat-form">
            <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); onSubmit(event); } }} placeholder={speechLanguage === "bn-BD" ? "প্রশ্ন লিখুন অথবা মাইক্রোফোনে বলুন…" : "Type a question or tap the microphone to speak…"} rows={2}/>
            <button type="submit" aria-label="বার্তা পাঠান" disabled={!input.trim() || loading}><Send size={19}/></button>
          </div>
          {(speechMessage || interimTranscript) && <p className={`voice-status ${listening ? "is-listening" : ""}`} aria-live="polite">
            {interimTranscript ? `${speechLanguage === "bn-BD" ? "শুনছি" : "Listening"}: ${interimTranscript}` : speechMessage}
          </p>}
        </form>
        <p className="voice-privacy-note">কথা লেখা ব্রাউজারের built-in speech recognition দিয়ে হয়—কোনো আলাদা API key বা অ্যাপ ফি নেই। কিছু ব্রাউজার অডিও তাদের speech service-এ পাঠাতে পারে; লেখা পাঠালে Krittim-S1 উত্তর তৈরির জন্য দূরবর্তী AI সেবায় পাঠানো হয়।</p>
        <p className="chat-disclaimer">AI-এর পরামর্শ সাধারণ তথ্য। রোগ, সার ও বালাই ব্যবস্থাপনায় স্থানীয় কৃষি বিশেষজ্ঞের পরামর্শ নিন।</p>
      </div>
    </section>
  </>;
}

function CropAdvice({ crop, districtName, setCrop, selectDistrict, guide }: { crop: string; districtName: string; setCrop: (value: string) => void; selectDistrict: (value: string) => void; guide: typeof cropGuides[string] }) {
  return <>
    <ScreenTitle eyebrow="স্থান ও ফসলভিত্তিক পরিকল্পনা" title="ফসল পরামর্শ"><div className="screen-filters"><DistrictSelect value={districtName} onChange={selectDistrict}/><CropSelect value={crop} onChange={setCrop}/></div></ScreenTitle>
    <section className="advice-hero"><div><p className="section-kicker">{districtName} জেলার জন্য নির্বাচন করা হয়েছে</p><h2>{crop} চাষের মৌলিক নির্দেশনা</h2><p>এটি সাধারণ, বাংলাদেশ-প্রাসঙ্গিক পরিচর্যা নির্দেশনা। জমির মাটি, জাত, মৌসুম এবং স্থানীয় পরিস্থিতি অনুযায়ী কৃষি সম্প্রসারণ কর্মকর্তার পরামর্শকে অগ্রাধিকার দিন।</p></div><div className="advice-crop-symbol"><Leaf size={69}/><span>{crop}</span></div></section>
    <section className="guide-grid"><article><CalendarDays size={21}/><p>সময় ও মৌসুম</p><strong>{guide.season}</strong></article><article><Map size={21}/><p>জমি ও মাটি</p><strong>{guide.soil}</strong></article><article><Droplets size={21}/><p>পানি ব্যবস্থাপনা</p><strong>{guide.water}</strong></article></section>
    <section className="practice-panel panel"><div><p className="section-kicker">ভালো চাষের ভিত্তি</p><h2>মাঠে যা খেয়াল রাখবেন</h2></div><div className="practice-list"><div><span>০১</span><p>{guide.note}</p></div><div><span>০২</span><p>জমির ইতিহাস, মাটির পরীক্ষা এবং অনুমোদিত জাতের তথ্য নিয়ে স্থানীয় কৃষি অফিসের সঙ্গে পরিকল্পনা মিলিয়ে নিন।</p></div><div><span>০৩</span><p>{guide.warning}</p></div></div></section>
    <section className="help-callout"><div className="help-icon"><MessageCircle size={22}/></div><div><h2>আপনার জমির অবস্থা আলাদা?</h2><p>মাটি, সেচ বা পাতার লক্ষণ জানিয়ে স্মার্ট কৃষি বুদ্ধিমত্তা AI-এর কাছে সাধারণ পরামর্শ নিতে পারেন।</p></div><button className="secondary-button" onClick={() => document.querySelector<HTMLButtonElement>(".nav-item:nth-child(6)")?.click()}>AI-কে প্রশ্ন করুন <ChevronRight size={17}/></button></section>
  </>;
}
