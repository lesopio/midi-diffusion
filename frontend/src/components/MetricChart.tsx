import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Filler);

export type MetricChartProps = {
  title: string;
  values: number[];
  color: string;
  windowSize?: number;
};

function computeRange(series: number[]) {
  if (!series.length) {
    return { min: 0, max: 1 };
  }
  let min = Math.min(...series);
  let max = Math.max(...series);
  if (min === max) {
    const pad = Math.abs(min) * 0.1 || 1;
    return { min: min - pad, max: max + pad };
  }
  const pad = (max - min) * 0.1;
  return { min: min - pad, max: max + pad };
}

export default function MetricChart({ title, values, color, windowSize = 200 }: MetricChartProps) {
  const series = values.filter(Number.isFinite).slice(-windowSize);
  const labels = series.map((_, idx) => idx + 1);
  const range = computeRange(series);
  return (
    <div className="chart-card">
      <div className="chart-title">{title}</div>
      <div className="chart-body">
        <Line
          data={{
            labels,
            datasets: [
              {
                data: series,
                borderColor: color,
                backgroundColor: "transparent",
                pointRadius: 0,
                borderWidth: 2,
                tension: 0.2,
                fill: false,
              },
            ],
          }}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
              x: {
                display: false,
              },
              y: {
                display: true,
                min: range.min,
                max: range.max,
                ticks: {
                  color: "#9aa0a6",
                  maxTicksLimit: 5,
                },
                grid: {
                  color: "rgba(255,255,255,0.06)",
                },
              },
            },
            plugins: {
              legend: { display: false },
            },
          }}
        />
      </div>
    </div>
  );
}
