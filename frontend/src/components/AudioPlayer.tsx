"use client";

import React, { useEffect, useRef, useState } from "react";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import { Play, Pause, Volume2 } from "lucide-react";

type Props = {
  src?: string | null;
  file?: File | null;
  className?: string;
};

export default function AudioPlayer({
  src = null,
  file = null,
  className = "",
}: Props) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);

  const finalSrc = src ?? objectUrl;

  useEffect(() => {
    if (file) {
      const u = URL.createObjectURL(file);
      setObjectUrl(u);
      return () => URL.revokeObjectURL(u);
    }
    return undefined;
  }, [file]);

  useEffect(() => {
    setCurrent(0);
    setPlaying(false);
    setDuration(0);
  }, [finalSrc]);

  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;

    const onTime = () => setCurrent(a.currentTime);
    const onLoaded = () => setDuration(a.duration || 0);
    const onEnded = () => setPlaying(false);

    a.addEventListener("timeupdate", onTime);
    a.addEventListener("loadedmetadata", onLoaded);
    a.addEventListener("ended", onEnded);

    return () => {
      a.removeEventListener("timeupdate", onTime);
      a.removeEventListener("loadedmetadata", onLoaded);
      a.removeEventListener("ended", onEnded);
    };
  }, [finalSrc]);

  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;
    a.volume = volume;
  }, [volume]);

  const togglePlay = () => {
    const a = audioRef.current;
    if (!a) return;
    if (a.paused) {
      void a.play().catch(() => {
        setPlaying(false);
      });
      setPlaying(true);
    } else {
      a.pause();
      setPlaying(false);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const a = audioRef.current;
    if (!a) return;
    const t = Number(e.target.value);
    a.currentTime = t;
    setCurrent(t);
  };

  const fmt = (s: number) => {
    if (!isFinite(s)) return "0:00";
    const mm = Math.floor(s / 60);
    const ss = Math.floor(s % 60)
      .toString()
      .padStart(2, "0");
    return `${mm}:${ss}`;
  };

  return (
    <Card className={`p-3 ${className}`}>
      <CardContent className="flex flex-col items-center gap-4 md:flex-row">
        <audio ref={audioRef} src={finalSrc ?? undefined} preload="metadata" />

        <div className="flex w-full flex-col items-center gap-3 sm:flex-row">
          <div className="flex-shrink-0">
            <Button
              variant="ghost"
              size="sm"
              onClick={togglePlay}
              aria-label={playing ? "Pause" : "Play"}
            >
              {playing ? (
                <Pause className="h-5 w-5" />
              ) : (
                <Play className="h-5 w-5" />
              )}
            </Button>
          </div>

          <div className="w-full md:w-64">
            <input
              aria-label="Seek"
              type="range"
              min={0}
              max={duration || 0}
              value={current}
              step={0.01}
              onChange={handleSeek}
              className="h-2 w-full rounded-lg accent-stone-600"
            />
            <div className="mt-1 flex justify-between text-xs text-stone-500">
              <span>{fmt(current)}</span>
              <span>{fmt(duration)}</span>
            </div>
          </div>

          <div className="flex w-full items-center gap-2 sm:w-auto">
            <Volume2 className="h-4 w-4 text-stone-500" />
            <input
              aria-label="Volume"
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={volume}
              onChange={(e) => setVolume(Number(e.target.value))}
              className="h-2 w-full rounded-lg sm:w-24 accent-black"
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
